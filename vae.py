import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from dataset import Dataset


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H):
        super(Encoder, self).__init__()
        # store patch size
        self.D_in = D_in
        self.out_features = 16

        self.conv1   = torch.nn.Conv2d(in_channels=1, out_channels=8,
                                       kernel_size=3, padding=1)
        self.conv2   = torch.nn.Conv2d(in_channels=8, out_channels=self.out_features,
                                       kernel_size=3, padding=1)
        self.linear1 = torch.nn.Linear(self.out_features * D_in * D_in, H)

    def forward(self, x):
        x = x.view(-1, 1, self.D_in, self.D_in)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, self.out_features * self.D_in * self.D_in)
        return F.relu(self.linear1(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(Decoder, self).__init__()
        # store patch size
        self.D_out = D_out
        self.in_features = 16

        self.linear1 = torch.nn.Linear(D_in, self.in_features * D_out * D_out)
        self.conv1   = torch.nn.Conv2d(in_channels=self.in_features, out_channels=8,
                                       kernel_size=3, padding=1)
        self.conv2   = torch.nn.Conv2d(in_channels=8, out_channels=1,
                                       kernel_size=3, padding=1)

    def forward(self, x):
        x = F.elu(self.linear1(x))
        x = x.view(-1, self.in_features, self.D_out, self.D_out)
        x = F.elu(self.conv1(x))
        return F.relu(self.conv2(x))


class VAE(torch.nn.Module):

    def __init__(self, input_dim, latent_dim=8, H=100):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, H)
        self.decoder = Decoder(latent_dim, input_dim)
        self._enc_mu = torch.nn.Linear(H, latent_dim)
        self._enc_log_sigma = torch.nn.Linear(H, latent_dim)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z.cuda(0), requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

if __name__ == '__main__':
    batch_size = 16
    patch_size = 64
    input_dim = patch_size

    dataset = Dataset(patch_size=patch_size, stride=32, size=6000, seed=1234)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=batch_size, shuffle=True)

    print('Number of samples:', len(dataset))

    vae_net = VAE(input_dim, latent_dim=8, H=100)

    criterion = nn.MSELoss()

    vae = nn.DataParallel(vae_net).cuda(0)
    criterion.cuda(0)

    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    l = None
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            batch = data[:,:,:,:,0]
            inputs = Variable(batch.cuda(0))
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            l = loss.item()
        print(epoch, l)
    torch.save(vae.state_dict(), 'vae_net_test.pth')
