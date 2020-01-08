from utils_deblur import psf2otf, otf2psf
import numpy as np


def deblurring_estimate(Y, X_l, k_l, reg_weight=1):
    '''
    Operation: solve for Z that minimizes: ||Y-k_l*Z||**2 + reg_weight * ||Z-X_l||**2
    Inputs: 
        2D images Y and X_l (Gray or multichannel)
        k_l (blur kernel for the low-res image, should be normalized to 1)
        reg_weight (weight of the reg term ||Z-X_l||**2)
    Outputs: 
        Z image that minimizes the optimization loss
    '''
    
    # Convert inputs to Fourier domain
    X_l_Freq = np.fft.fft2(X_l, axes=[0, 1])
    
    Y_Freq = np.fft.fft2(Y, axes=[0, 1])
    
    k_l_Freq = psf2otf(k_l, Y.shape[:2])

    if X_l_Freq.ndim == 3:
        k_l_Freq = np.repeat(k_l_Freq[:, :, np.newaxis], X_l_Freq.shape[2], axis=2)
    
    
    # Solve for k in Fourier domain (regularization only affects den)
    num = k_l_Freq.conjugate() * Y_Freq + reg_weight * X_l_Freq
    den = np.abs(k_l_Freq)**2 + reg_weight # Fourier transform of k_l transpose * k_l + reg_weight
    
    Z_Freq = num / den

    # Convert back to spatial, given the width
    Z = np.real(np.fft.ifft2(Z_Freq, axes=(0, 1)))

    return Z


