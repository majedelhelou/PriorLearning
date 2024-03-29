import torch
import torch.nn as nn
import os
from util import Kernels

class CNN_Model(nn.Module):
    def __init__(self, num_of_layers=10, kernel_size=3, padding=1, features=64, gksize=11, gsigma=3):
        super(CNN_Model, self).__init__()
        # We are only interested in grayscale
        channels = 1
        # Padding to keep img size constant
        padding = (kernel_size - 1) // 2
        # Create the gaussian blur kernel
        gkernel = Kernels.kernel_2d(gksize, gsigma)

        def Conv2d_train(in_channels, out_channels):
            '''
            Utility method to create a conv2d layer
            '''

            return nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                padding      = padding,
                bias         = False
            )

        def Conv2d_blur(kernel):
            '''
            Utility method to create the gaussian blur layer
            '''

            padding = (gksize - 1) // 2
            layer = nn.Conv2d(
                in_channels  = channels,
                out_channels = channels,
                kernel_size  = kernel.shape,
                padding      = padding,
                bias         = False
            )

            # Set the kernel to the gaussian blur kernel
            w, h = kernel.shape
            gaussian_kernel = torch.ones(1, 1, w, h)
            gaussian_kernel[0, 0] = torch.from_numpy(kernel)
            layer.weight = torch.nn.Parameter(gaussian_kernel)

            return layer

        layers = []
        layers.append(Conv2d_train(channels, features))
        layers.append(nn.ReLU(inplace=True))

        # Add the specified number of layers (conv2d and batch normalization with relu activations)
        for _ in range(num_of_layers-2):
            layers.append(Conv2d_train(features, features))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(Conv2d_train(features, channels))

        layers.append(Conv2d_blur(gkernel))

        self.network = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        if 'deblurr' not in kwargs:
            deblurr = False
        else:
            deblurr = kwargs['deblurr']

        if deblurr:
            out = self.network[:-1](x)
        else:
            out = self.network(x)

        return out
