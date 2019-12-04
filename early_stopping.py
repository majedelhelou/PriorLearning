import os
import numpy as np
import torch

class EarlyStopping:
    def __init__(self, dataset_size, seed, ps, stride, lr, layers, kernel_size, features, patience=10):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.min_val_loss = np.Inf
        self.dataset_size = dataset_size
        self.model_name = 'DSseed%d_ps%d_stride%d_lr%d_layers%d_kernel%d_features%d' \
                           % (seed, ps, stride, lr, layers, kernel_size, features)

    def __call__(self, val_loss, model):
        if val_loss >= min_val_loss:
            self.counter += 1
            print('No improvement for %d epochs' %self.counter)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        model_dir = os.path.join(self.model_name, 'DSsize%d' % self.dataset_size)
        model_dir = os.path.join('saved_models', model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, 'net.pth'))
        self.min_val_loss = val_loss