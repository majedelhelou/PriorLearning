import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import CNN_Model
from dataset import prepare_data, Dataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="Deblurring")

parser.add_argument("--preprocess",          type=bool,  default=False, help='run prepare_data or not')
parser.add_argument("--patch_size",          type=int,   default=32,    help='dataset patch size')
parser.add_argument("--stride",              type=int,   default=16,    help='dataset stride')
parser.add_argument("--gksize",              type=int,   default=11,    help='blur kernel size')
parser.add_argument("--gsigma",              type=int,   default=3,     help='blur kernel sigma')
parser.add_argument("--dataset_size",        type=int,   default=100,   help='train dataset size to use')
parser.add_argument("--dataset_seed",        type=int,   default=1234,  help='seed to determine dataset order')
parser.add_argument("--network_kernel_size", type=int,   default=3,     help='network convolution kernel size')
parser.add_argument("--network_features",    type=int,   default=64,    help='network numbre of features')
parser.add_argument("--batch_size",          type=int,   default=100,   help="Training batch size")
parser.add_argument("--num_of_layers",       type=int,   default=10,    help="Number of total layers")
parser.add_argument("--epochs",              type=int,   default=50,    help="Number of training epochs")
parser.add_argument("--milestone",           type=int,   default=30,    help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr",                  type=float, default=1e-3,  help="Initial learning rate")

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

    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    print('** Creating network: **')
    net = CNN_Model(
        num_of_layers = opt.num_of_layers,
        kernel_size   = opt.network_kernel_size,
        features      = opt.network_features,
        gksize        = opt.gksize,
        gsigma        = opt.gsigma
    )

    # Loss
    criterion = nn.MSELoss(size_average=False)

    # Move to GPU
    model = nn.DataParallel(net).cuda()
    criterion.cuda()

    # Optimizer
    # TODO: check if that's correct and doesn't train the fixed convolution layer
    optimizer = optim.Adam(model.network[:-1].parameters(), lr=opt.lr)

    train_loss_log = np.zeros(opt.epochs)

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
        for i, data in enumerate(loader_train, 0):
            # Training
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # Training step
            img_train = Variable(data.cuda())

            out_train = model(img_train)
            loss = criterion(out_train, img_train) / (img_train.size()[0]*2)

            loss.backward()
            optimizer.step()

            train_loss_log[epoch] += loss.item()

        train_loss_log[epoch] = train_loss_log[epoch] / len(loader_train)

        print('Epoch %d: loss=%.4f' %(epoch, train_loss_log[epoch]))

        # TODO Remove when sure last layer isn't trained
        print(model.network[-1].weight)

        model_name = 'DSseed%d_ps%_stride%s_lr%d_layers%d_kernel%d_features%d' % (
            opt.dataset_seed,
            opt.patch_size,
            opt.stride,
            opt.lr,
            opt.num_of_layers,
            opt.network_kernel_size,
            opt.network_features
        )

        model_name_with_size = os.path.join(model_name, 'DSsize%d' % opt.dataset_size)

        model_dir = os.path.join('saved_models', model_name_with_size)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, 'net_%d.pth' % (epoch)) )


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(
            patch_size = opt.patch_size,
            stride     = opt.stride,
            gksize     = opt.gksize,
            gsigma     = opt.gsigma
        )

    main()
