import os
import cv2
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import CNN_Model
from util import Kernels, batch_PSNR


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="DeblurringTest")

parser.add_argument("--model_seed",        type=int,    default=1234, help='seed used when training the model')
parser.add_argument("--model_patch_size",  type=int,    default=64,   help='train dataset patch size')
parser.add_argument("--model_stride",      type=int,    default=32,   help='train dataset stride')
parser.add_argument("--model_lr",          type=float,  default=1e-3, help='lr used for training the model')
parser.add_argument("--model_num_layers",  type=int,    default=10,   help='number of layers in the model')
parser.add_argument("--model_kernel_size", type=int,    default=3,    help='kernel size in the model')
parser.add_argument("--model_features",    type=int,    default=64,   help='number of features in the model')
parser.add_argument("--gksize",            type=int,    default=11,   help='blur kernel size')
parser.add_argument("--gsigma",            type=int,    default=3,    help='blur kernel sigma')

opt = parser.parse_args()


def normalize(data):
    return data/255.


def inference(test_data, model):
    files_source = glob.glob(os.path.join(test_data, 'BSD68', '*.png'))

    files_source.sort()
    kernel = Kernels.kernel_2d(opt.gksize, opt.gsigma)
    psnr_results = []

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
            psnr_results.append(batch_PSNR(IOut, ISource, 1.))

    return np.mean(psnr_results)


def main():

    base_to_patch = ((180 - opt.model_patch_size) // opt.model_stride + 1) ** 2

    model_name = 'DSseed%d_ps%d_stride%d_lr%d_layers%d_kernel%d_features%d' % (
        opt.model_seed,
        opt.model_patch_size,
        opt.model_stride,
        opt.model_lr,
        opt.model_num_layers,
        opt.model_kernel_size,
        opt.model_features
    )

    log_dir = os.path.join('logs', model_name)
    model_dir = os.path.join('saved_models', model_name)

    model_DSsizes = os.listdir(model_dir)

    # There are 400 base train images
    nb_base_train_images = 400
    results = np.zeros(nb_base_train_images)

    for model_DSsize in model_DSsizes:
        DSsize = int(model_DSsize.split('size')[-1])
        size_idx = (DSsize // base_to_patch) - 1

        trained_dir = os.path.join(model_dir, model_DSsize)
        model_trained = os.listdir(trained_dir)[0]

        print('Testing size_idx %d' % size_idx)

        net = CNN_Model(
            num_of_layers = opt.model_num_layers,
            kernel_size   = opt.model_kernel_size,
            features      = opt.model_features,
            gksize        = opt.gksize,
            gsigma        = opt.gsigma
        )

        model = nn.DataParallel(net).cuda(0)
        model.load_state_dict(torch.load(os.path.join(trained_dir, model_trained)))
        model.eval()

        mean_psnr = inference('data', model)

        print('Mean_psnr:', mean_psnr)

        results[size_idx] = mean_psnr


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    np.save(os.path.join(log_dir, 'results'), results)

    print('Results saved inside Logs!')


if __name__ == "__main__":
    main()
