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

parser.add_argument("--model_seed",        type=int,    default=1234,  help='seed used when training the model')
parser.add_argument("--model_patch_size",  type=int,    default=64,    help='train dataset patch size')
parser.add_argument("--model_stride",      type=int,    default=32,    help='train dataset stride')
parser.add_argument("--model_lr",          type=float,  default=1e-3,  help='lr used for training the model')
parser.add_argument("--model_num_layers",  type=int,    default=10,    help='number of layers in the model')
parser.add_argument("--model_kernel_size", type=int,    default=3,     help='kernel size in the model')
parser.add_argument("--model_features",    type=int,    default=64,    help='number of features in the model')
parser.add_argument("--gksize",            type=int,    default=11,    help='blur kernel size')
parser.add_argument("--gsigma",            type=int,    default=3,     help='blur kernel sigma')
parser.add_argument("--optimizer",         type=str,    default='SGD', help="Network optimizer")
parser.add_argument("--batch_size",        type=int,    default=16,    help="Training batch size")
parser.add_argument("--augmentation",      type=str,    default='no',  help="dataset augmentation")
parser.add_argument("--num_images",        type=int,    default=5,     help="number of images to save")

opt = parser.parse_args()


def normalize(data):
    return data / 255.

def denormalize(data):
    return np.clip(data, 0., 1.) * 255.

def save_imgs(test_data, model, target_folder):
    '''
    Save the clear and blurred img (in) and the clear and blurred img (out)
    Inputs:
        test_data: folder where the data in stored
        model: trained network
        target_folder: where to save the images
    '''

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    files_source = glob.glob(os.path.join(test_data, 'BSD68', '*.png'))

    files_source.sort()
    # Create the gausian blur kernel
    kernel = Kernels.kernel_2d(opt.gksize, opt.gsigma)

    for f in files_source[:opt.num_images]:
        # Read the clear img
        Img_clear = cv2.imread(f)

        # Blur, normalize and transfer it to the GPU
        Img_blurred = cv2.filter2D(np.float32(Img_clear), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        Img = normalize(np.float32(Img_blurred[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        ISource = Variable(ISource.cuda(0))

        with torch.no_grad():
            # Get the clear and blurred img (out) from the network
            IOut = model(ISource)
            IOut_clear = model(ISource, deblurr=True)

        # denormalize and convert to 8-bit integers before saving
        Img_blurred = np.uint8(Img_blurred)

        IOut = IOut.data.cpu().numpy().astype(np.float32)
        IOut = np.uint8(denormalize(IOut))
        _, _, h, w = IOut.shape
        IOut = IOut.reshape((h, w))

        IOut_clear = IOut_clear.data.cpu().numpy().astype(np.float32)
        IOut_clear = np.uint8(denormalize(IOut_clear))
        _, _, h, w = IOut_clear.shape
        IOut_clear = IOut_clear.reshape((h, w))

        base_name = f.split('/')[-1].split('.')[0]

        # Save the images
        cv2.imwrite(os.path.join(target_folder, base_name + '_in_clear.png'), Img_clear)
        cv2.imwrite(os.path.join(target_folder, base_name + '_in_blurred.png'), Img_blurred)
        cv2.imwrite(os.path.join(target_folder, base_name + '_out_clear.png'), IOut_clear)
        cv2.imwrite(os.path.join(target_folder, base_name + '_out_blurred.png'), IOut)

        Img_clear = Img_clear[:,:,0]
        Img_blurred = Img_blurred[:,:,0]

        clear_both = np.hstack((Img_clear, IOut_clear))
        blur_both  = np.hstack((Img_blurred, IOut))
        grid       = np.vstack((clear_both, blur_both))

        # Save the grid of the four images
        cv2.imwrite(os.path.join(target_folder, base_name + '_grid.png'), grid)


def main():

    # For printing purposes
    base_to_patch = ((180 - opt.model_patch_size) // opt.model_stride + 1) ** 2

    model_name = 'DSseed%d_%s_lr%s_batchsize%d_depth%d_gsigma%d' % (
        opt.model_seed,
        opt.optimizer,
        'p'.join(str(opt.model_lr).split('.')),
        opt.batch_size,
        opt.model_num_layers,
        opt.gsigma
    )

    if opt.augmentation == 'standard':
        model_name += '_augmented'
    elif opt.augmentation == 'vae':
        model_name += 'vae'

    img_dir = os.path.join('saved_images', model_name)
    model_dir = os.path.join('saved_models', model_name)

    model_DSsizes = os.listdir(model_dir)

    # There are 400 base train images
    nb_base_train_images = 400
    results = np.zeros(nb_base_train_images)

    # Save images for each dataset size
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

        # Load the trained model state
        model_path = os.path.join(trained_dir, model_trained)
        print('Loading model:', model_path)

        model = nn.DataParallel(net).cuda(0)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        save_imgs('data', model, os.path.join(img_dir, model_DSsize))


if __name__ == "__main__":
    main()
