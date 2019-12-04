import os
import argparse
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import CNN_Model
from dataset import prepare_data, Dataset
from early_stopping import EarlyStopping


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="Deblurring")

parser.add_argument("--preprocess",          type=bool,   default=False, help='run prepare_data or not')
parser.add_argument("--patch_size",          type=int,    default=64,    help='dataset patch size')
parser.add_argument("--stride",              type=int,    default=32,    help='dataset stride')
parser.add_argument("--gksize",              type=int,    default=11,    help='blur kernel size')
parser.add_argument("--gsigma",              type=int,    default=3,     help='blur kernel sigma')
parser.add_argument("--dataset_size",        type=int,    default=16,    help='train dataset size to use')
parser.add_argument("--dataset_seed",        type=int,    default=1234,  help='seed to determine dataset order')
parser.add_argument("--network_kernel_size", type=int,    default=3,     help='network convolution kernel size')
parser.add_argument("--network_features",    type=int,    default=64,    help='network numbre of features')
parser.add_argument("--batch_size",          type=int,    default=16,    help="Training batch size")
parser.add_argument("--num_of_layers",       type=int,    default=10,    help="Number of total layers")
parser.add_argument("--epochs",              type=int,    default=50,    help="Number of training epochs")
parser.add_argument("--milestone",           type=int,    default=30,    help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr",                  type=float,  default=1e-3,  help="Initial learning rate")
parser.add_argument("--optimizer",           type=string, default='SGD', help="Network optimizer")

opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')

    dataset_train = Dataset(
        patch_size = opt.patch_size,
        stride     = opt.stride,
        size       = opt.dataset_size,
        seed       = opt.dataset_seed
    )

    loader_train = DataLoader(
        dataset=dataset_train,
        num_workers=4,
        batch_size=opt.batch_size,
        shuffle=True
    )

    print("# of training samples: %d\n" % num_train_examples)

    print('** Creating network: **')
    net = CNN_Model(
        num_of_layers = opt.num_of_layers,
        kernel_size   = opt.network_kernel_size,
        features      = opt.network_features,
        gksize        = opt.gksize,
        gsigma        = opt.gsigma
    )

    non_trainable_layer_idx = str(len(net.network) - 1)

    # Loss
    criterion = nn.MSELoss(size_average=False)

    # Move to GPU
    model = nn.DataParallel(net).cuda(0)
    criterion.cuda(0)

    # Set last layer to non trainable
    for name, param in model.named_parameters():
        if non_trainable_layer_idx in name:
            param.requires_grad = False

    # Optimizer
    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    train_loss_log = np.zeros(opt.epochs)
    validation_loss_log = np.zeros(opt.epochs)

    early_stopping = EarlyStopping(
        opt.dataset_size,
        opt.dataset_seed,
        opt.patch_size,
        opt.stride,
        opt.lr,
        opt.num_of_layers,
        opt.network_kernel_size,
        opt.network_features
    )

    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / (10.)
        # Learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('\nlearning rate %f' % current_lr)

        # Train
        model.train()
        for i, data in enumerate(loader_train, 0):
            model.zero_grad()
            optimizer.zero_grad()
            batch = data[:,:,:,:,1]

            # Training step
            img_train = Variable(batch.cuda(0))
            out_train = model(img_train)

            loss = criterion(out_train, img_train) / (img_train.size()[0]*2)

            loss.backward()
            optimizer.step()

            train_loss_log[epoch] += loss.item()

        train_loss_log[epoch] = train_loss_log[epoch] / len(loader_train)

        # Eval
        model.eval()
        files_source = glob.glob(os.path.join(test_data, 'BSD68', '*.png'))
        files_source.sort()
        kernel = Kernels.kernel_2d(opt.gksize, opt.gsigma)
        for f in files_source:
            Img = cv2.imread(f)
            Img = cv2.filter2D(np.float32(Img), -1, kernel, borderType=cv2.BORDER_CONSTANT)
            Img = normalize(np.float32(Img[:,:,0]))
            Img = np.expand_dims(Img, 0)
            Img = np.expand_dims(Img, 1)
            ISource = torch.Tensor(Img)
            ISource = Variable(ISource.cuda(0))
            with torch.no_grad():
                IOut = model(ISource)
                loss = criterion(IOut, ISource) / (ISource.size()[0]*2)
                validation_loss_log[epoch] += loss.item()

        validation_loss_log[epoch] = validation_loss_log[epoch] / len(files_source)

        # TODO get training and validation loss on ground truth images
        model_ground_truth = nn.Sequential(*list(model.children())[:-1])

        print('Epoch %d: train_loss=%.4f, validation_loss=%.4f' \
               %(epoch, train_loss_log[epoch], validation_loss_log[epoch]))

        early_stopping(validation_loss_log[epoch], model)

        if early_stopping.early_stop:
            print('Early stopping triggered.')

        model_dir = os.path.join(early_stopping.model_name, 'DSsize%d' % opt.dataset_size)
        model_dir = os.path.join('saved_models', model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        np.save(os.path.join(model_dir, 'train_loss'), train_loss_log)
        np.save(os.path.join(model_dir, 'validation_loss'), validation_loss_log)


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(
            patch_size = opt.patch_size,
            stride     = opt.stride,
            gksize     = opt.gksize,
            gsigma     = opt.gsigma
        )

    main()
