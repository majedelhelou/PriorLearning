import os
import torch
import torch.utils.data
import cv2
import h5py
import numpy as np
from util import Kernels
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from dataset import Dataset

# changed configuration to this instead of argparse for easier interaction
CUDA = True
SEED = 1234
BATCH_SIZE = 16
LOG_INTERVAL = 10
EPOCHS = 10
no_of_sample = 10
patch_size = 64

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}

# Use full default dataset for training
dataset_train = Dataset(patch_size=patch_size, stride=32, size=6400, seed=SEED)
# shuffle data at every epoch
train_loader = torch.utils.data.DataLoader(
    dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs
)

dataset_test = Dataset(patch_size=patch_size, stride=32, size=640, seed=SEED)
# Same for test data
test_loader = torch.utils.data.DataLoader(
    dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True, **kwargs
)


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, pool_kernel_size=(2, 2)):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding, stride)
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool(x)

        return x


class VAE(nn.Module):
    def __init__(self, patch_size):
        super(VAE, self).__init__()

        self.patch_size = patch_size

        # Encoder
        self.block1 = Conv_Block(1, 64, (3, 3), 1, 1)  # 32
        self.block2 = Conv_Block(64, 128, (3, 3), 1, 1)  # 16
        self.block3 = Conv_Block(128, 32, (3, 3), 1, 1)  # 8

        # Decoder
        self.fct_decode = nn.Sequential(
            nn.Conv2d(16, 64, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32
            nn.Conv2d(64, 16, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64
        )

        self.final_decod_mean = nn.Conv2d(16, 1, (3, 3), padding=1)


    def encode(self, x: Variable) -> (Variable, Variable):
        x = x.view(-1, 1, self.patch_size, self.patch_size)
        x = F.elu(self.block1(x))
        x = F.elu(self.block2(x))
        x = F.elu(self.block3(x))

        return x[:, :16, :, :], x[:, 16:, :, :]  # output shape - batch_size x 16 x 8 x 8


    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation

            sample_z = []
            for _ in range(no_of_sample):
                std = logvar.mul(0.5).exp_()  # type: Variable
                eps = Variable(std.data.new(std.size()).normal_())
                sample_z.append(eps.mul(std).add_(mu))

            return sample_z

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu


    def decode(self, z: Variable) -> Variable:
        z = self.fct_decode(z)
        z = self.final_decod_mean(z)
        z = F.sigmoid(z)

        return z.view(-1, self.patch_size * self.patch_size)


    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, self.patch_size * self.patch_size))
        z = self.reparameterize(mu, logvar)
        if self.training:
            return [self.decode(z) for z in z], mu, logvar
        else:
            return self.decode(z), mu, logvar


    def loss_function(self, recon_x, x, mu, logvar) -> Variable:
        # how well do input x and output recon_x agree?

        if self.training:
            BCE = 0
            for recon_x_one in recon_x:
                BCE += F.binary_cross_entropy(recon_x_one, x.view(-1, self.patch_size * self.patch_size))
            BCE /= len(recon_x)
        else:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.patch_size * self.patch_size))

        # KLD is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= BATCH_SIZE * self.patch_size * self.patch_size

        return BCE + KLD


model = VAE(patch_size)
if CUDA:
    model.cuda(0)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch, train_loader=train_loader):
    # toggle model to train mode
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = Variable(data[:,:,:,:,0])
        if CUDA:
            data = data.cuda(0)
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(data)
        # calculate scalar loss
        loss = model.loss_function(recon_batch, data, mu, logvar)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0

    # each data is of BATCH_SIZE (default 16) samples
    for i, data in enumerate(test_loader):
        if CUDA:
            # make sure this lives on the GPU
            data = data.cuda(0)

        # we're only going to infer, so no autograd at all required: volatile=True
        data = Variable(data[:,:,:,:,0], volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += model.loss_function(recon_batch, data, mu, logvar).item()
        if i == 0:
            n = min(data.size(0), 8)
            # for the first 128 batch of the epoch, show the first 8 input digits
            # with right below them the reconstructed output digits
            comparison = torch.cat([data[:n],
                                    recon_batch.view(BATCH_SIZE, 1, patch_size, patch_size)[:n]])
            save_image(comparison.data.cpu(),
                       './vae_generated/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def augment_with_vae(patch_size=64, stride=32, size='full', seed=1234, gksize=11, gsigma=3):
    '''
    Create the augmented dataset and save it
    Inputs:
        patch_size: the patch_size of the dataset to augment
        stride: the stride of the dataset to augment
        size: the size of the dataset to use for vae training
        seed: the seed of the dataset to use for vae training
        gksize: the gaussian blur kernel size
        gsigma: the gaussian blur kernel sigma parameter
    '''

    print('agumenting dataset with vae')

    # Load the dataset to use for training
    ds = Dataset(patch_size=patch_size, stride=stride, size=size, seed=seed)
    loader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=16,
        shuffle=True,
        **kwargs
    )

    # Train the VAE
    for epoch in range(1, EPOCHS + 1):
        train(epoch, train_loader=loader)

    # Augment and save the dataset
    augmented_file_name = 'datasets/train_ps%d_stride%d_vae.h5' % (patch_size, stride)
    h5f = h5py.File(augmented_file_name, 'w')

    # Create the gaussian blur kernel
    kernel = Kernels.kernel_2d(gksize, gsigma)

    loader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle=False
    )

    for patch_num, patch in enumerate(loader):
        patch = patch[0].numpy()
        h5f.create_dataset(str(patch_num), data=patch)

    # To be fair with the standard data augmentation method, we create 7 times the number
    # of images in the train set to augment it
    for i in range(7 * len(loader)):
        sample = Variable(torch.randn(1, 16, 8, 8))
        if CUDA:
            sample = sample.cuda(0)
        sample = model.decode(sample).cpu().detach().numpy()
        new_patch = sample.reshape(1, patch_size, patch_size, 1)
        blurred = np.expand_dims(
            np.expand_dims(
                cv2.filter2D(new_patch[0], -1, kernel, borderType=cv2.BORDER_CONSTANT),
                -1
            ),
            0
        )
        data = np.concatenate((new_patch, blurred), axis=-1)
        h5f.create_dataset(str(len(loader) + i), data=data)

    h5f.close()
    print('dataset augmentation with vae done')

if __name__ == "__main__":

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)

        # 64 sets of random ZDIMS-float vectors, i.e. 64 locations
        sample = Variable(torch.randn(64, 16, 8, 8))
        if CUDA:
            sample = sample.cuda(0)
        sample = model.decode(sample).cpu()

        # save out as an 8x8 matrix of images
        # this will give you a visual idea of how well latent space can generate things
        save_image(sample.data.view(64, 1, patch_size, patch_size),'./vae_generated/reconstruction' + str(epoch) + '.png')
