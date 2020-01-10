import os
import cv2
import argparse
import glob
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
from util import Kernels
from utils_fourier import deblurring_estimate

parser = argparse.ArgumentParser(description="Baseline")

parser.add_argument("--gksize",       type=int,   default=11, help='blur kernel size')
parser.add_argument("--gsigma",       type=int,   default=3,  help='blur kernel sigma')
parser.add_argument("--nb_img_saved", type=int,   default=5,  help='Number of images to save')
parser.add_argument("--reg_weight",   type=float, default=1e-2, help='L2 reg weight')

opt = parser.parse_args()

def normalize(data):
    return data / 255.

def denormalize(data):
    return np.clip(data, 0., 1.) * 255.

def main():
    kernel = Kernels.kernel_2d(opt.gksize, opt.gsigma)
    pad = (opt.gksize - 1) // 2

    files_source = glob.glob(os.path.join('data', 'BSD68', '*.png'))
    files_source.sort()

    target_folder = os.path.join('saved_images', 'baseline_gksize%d_gsigma%d' % (opt.gksize, opt.gsigma))
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    losses = []
    for idx, f in enumerate(files_source):
        # Load and blur the image
        Img_clear = cv2.imread(f)[:,:,0]
        #I_zero = np.zeros(Img_clear.shape)
        Img_blurred = cv2.filter2D(np.float32(Img_clear), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        Img_blurred = normalize(np.float32(Img_blurred))
        Img_clear = normalize(np.float32(Img_clear))

        # Unblur the image
        IOut_clear = deblurring_estimate(Img_blurred, Img_blurred, kernel, opt.reg_weight)

        loss = compare_psnr(IOut_clear[pad:-pad, pad:-pad], Img_clear[pad:-pad, pad:-pad], data_range=1.)
        losses.append(loss)

        if idx < opt.nb_img_saved:
            base_name = f.split('/')[-1].split('.')[0]

            # Convert back to 8-bit grayscale
            I_clear = np.uint8(denormalize(Img_clear))
            I_blurred = np.uint8(denormalize(Img_blurred))
            Out_clear = np.uint8(denormalize(IOut_clear))

            cv2.imwrite(os.path.join(target_folder, base_name + '_in_clear.png'), I_clear)
            cv2.imwrite(os.path.join(target_folder, base_name + '_in_blurred.png'), I_blurred)
            cv2.imwrite(os.path.join(target_folder, base_name + '_out_clear.png'), Out_clear)

            # Also save the three images side by side
            grid = np.hstack((I_clear, I_blurred, Out_clear))

            cv2.imwrite(os.path.join(target_folder, base_name + '_grid.png'), grid)

    # Just print the mean PSNR as it's not that useful to create a file to store a single number in this case
    print('Mean PSNR is', np.mean(losses))


if __name__ == "__main__":
    main()
