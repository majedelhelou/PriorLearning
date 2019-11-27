import numpy as np
import torch
import math
#from skimage.measure.simple_metrics import compare_psnr


class Kernels:

    @staticmethod
    def kernel_2d(
        kernel_size,
        sigma
    ):
        center = (kernel_size - 1) / 2
        kernel = np.fromfunction(
            lambda x, y:
                np.exp( -0.5 * ((x - center) ** 2 + (y - center) ** 2) / (sigma ** 2)),
            (kernel_size, kernel_size),
            dtype=float
        )
        kernel = kernel / np.sum(kernel)
        return kernel


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        mse = np.mean(np.square(Iclean[i,:,:,:] - Img[I,:,:,:]))
        psnr = 10 * np.log10((data_range ** 2) / mse)
        PSNR += psnr
        #PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    if math.isnan(PSNR):
        import pdb; pdb.set_trace()
    return (PSNR/Img.shape[0])
