import os
import cv2
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import CNN_Model
from dataset import normalize, prepare_data, Dataset, augment_dataset
from early_stopping import EarlyStopping
from util import Kernels, batch_PSNR, batch_MSE
from vae import augment_with_vae


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
parser.add_argument("--epochs",              type=int,    default=150,   help="Number of training epochs")
parser.add_argument("--milestone",           type=int,    default=50,    help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr",                  type=float,  default=1e-3,  help="Initial learning rate")
parser.add_argument("--optimizer",           type=str,    default='SGD', help="Network optimizer")
parser.add_argument("--augmentation",        type=str,    default='no',  help="dataset augmentation")

opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')

    mode = None
    ds_size = opt.dataset_size
    if opt.augmentation == 'standard':
        mode = 'augmented'
        ds_size = 'full'
    elif opt.augmentation == 'vae':
        mode = 'vae'
        ds_size = 'full'

    dataset_train = Dataset(
        patch_size = opt.patch_size,
        stride     = opt.stride,
        size       = ds_size,
        seed       = opt.dataset_seed,
        mode       = mode
    )

    loader_train = DataLoader(
        dataset=dataset_train,
        num_workers=4,
        batch_size=opt.batch_size,
        shuffle=True
    )

    print("# of training samples: %d\n" % int(len(dataset_train)))

    print('** Creating network: **')
    net = CNN_Model(
        num_of_layers = opt.num_of_layers,
        kernel_size   = opt.network_kernel_size,
        features      = opt.network_features,
        gksize        = opt.gksize,
        gsigma        = opt.gsigma
    )

    # To disable learning of the gaussian blur layer
    non_trainable_layer_idx = str(len(net.network) - 1)

    # Loss
    criterion = nn.MSELoss()

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

    # Create empty arrays to store the metrics at each epoch
    train_loss_log = np.zeros(opt.epochs)
    validation_loss_log = np.zeros(opt.epochs)
    validation_psnr_log = np.zeros(opt.epochs)
    validation_loss_clear = np.zeros(opt.epochs)
    validation_psnr_clear = np.zeros(opt.epochs)

    # Initialize the early stopping
    early_stopping = EarlyStopping(
        opt.dataset_size,
        opt.optimizer,
        opt.dataset_seed,
        opt.lr,
        opt.batch_size,
        opt.num_of_layers,
        opt.gsigma,
        augmentation=opt.augmentation
    )

    # border size to exclude (cropping)
    pad = (opt.gksize - 1) // 2

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

            loss = criterion(out_train, img_train)

            loss.backward()
            optimizer.step()

            train_loss_log[epoch] += loss.item()

        train_loss_log[epoch] = train_loss_log[epoch] / len(loader_train)

        # Eval on the validation set
        model.eval()
        files_source = glob.glob(os.path.join('data', 'BSD68', '*.png'))
        files_source.sort()
        kernel = Kernels.kernel_2d(opt.gksize, opt.gsigma)
        for f in files_source:
            # Load the img
            Img_clear = cv2.imread(f)

            # Blur the img and put on GPU
            Img = cv2.filter2D(np.float32(Img_clear), -1, kernel, borderType=cv2.BORDER_CONSTANT)
            Img = normalize(np.float32(Img[:,:,0]))
            Img = np.expand_dims(Img, 0)
            Img = np.expand_dims(Img, 1)
            ISource = torch.Tensor(Img)
            ISource = Variable(ISource.cuda(0))

            # Normalize the clear img and put on GPU
            Img_clear = normalize(np.float32(Img_clear[:,:,0]))
            Img_clear = np.expand_dims(Img_clear, 0)
            Img_clear = np.expand_dims(Img_clear, 1)
            ISource_clear = torch.Tensor(Img_clear)
            ISource_clear = Variable(ISource_clear.cuda(0))
            with torch.no_grad():
                # Get the blurred img (out) from the network
                IOut = model(ISource)
                loss = criterion(IOut[:,:,pad:-pad,pad:-pad], ISource[:,:,pad:-pad,pad:-pad])
                validation_loss_log[epoch] += loss.item()
                validation_psnr_log[epoch] += batch_PSNR(IOut[:,:,pad:-pad,pad:-pad], ISource[:,:,pad:-pad,pad:-pad], 1.)

                # Get the clear img (out) before gaussian blur layer
                IOut_clear = model(ISource, deblurr=True)
                loss_clear = criterion(IOut_clear[:,:,pad:-pad,pad:-pad], ISource_clear[:,:,pad:-pad,pad:-pad])
                validation_loss_clear[epoch] += loss_clear.item()
                validation_psnr_clear[epoch] += batch_PSNR(IOut_clear[:,:,pad:-pad,pad:-pad], ISource_clear[:,:,pad:-pad,pad:-pad], 1.)

        # Update the metrics
        validation_loss_log[epoch] = validation_loss_log[epoch] / len(files_source)
        validation_psnr_log[epoch] = validation_psnr_log[epoch] / len(files_source)
        validation_loss_clear[epoch] = validation_loss_clear[epoch] / len(files_source)
        validation_psnr_clear[epoch] = validation_psnr_clear[epoch] / len(files_source)

        print('Epoch %d: train_loss=%.4f, validation_loss=%.4f, validation_psnr=%.4f, val_GT_loss=%.4f, val_GT_psnr=%.4f' \
               %(epoch, train_loss_log[epoch], validation_loss_log[epoch], validation_psnr_log[epoch],
                 validation_loss_clear[epoch], validation_psnr_clear[epoch]))

        early_stopping(validation_loss_log[epoch], model)

        if early_stopping.early_stop:
            print('Early stopping triggered.')
            break

    # Save the results in files
    log_dir = os.path.join(early_stopping.model_name, 'DSsize%d' % opt.dataset_size)
    log_dir = os.path.join('logs', log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    np.save(os.path.join(log_dir, 'train_loss'), train_loss_log)
    np.save(os.path.join(log_dir, 'validation_loss'), validation_loss_log)
    np.save(os.path.join(log_dir, 'validation_psnr'), validation_psnr_log)
    np.save(os.path.join(log_dir, 'val_GT_loss'), validation_loss_clear)
    np.save(os.path.join(log_dir, 'val_GT_psnr'), validation_psnr_clear)


if __name__ == "__main__":
    # Create the training set
    if opt.preprocess:
        prepare_data(
            patch_size = opt.patch_size,
            stride     = opt.stride,
            gksize     = opt.gksize,
            gsigma     = opt.gsigma
        )

    # Perform some data augmentation on the training set
    if opt.augmentation=='standard':
        augment_dataset(
            patch_size = opt.patch_size,
            stride     = opt.stride,
            size       = opt.dataset_size,
            seed       = opt.dataset_seed
        )
    elif opt.augmentation=='vae':
        augment_with_vae(
            patch_size = opt.patch_size,
            stride     = opt.stride,
            size       = opt.dataset_size,
            seed       = opt.dataset_seed,
            gksize     = opt.gksize,
            gsigma     = opt.gsigma
        )

    main()
