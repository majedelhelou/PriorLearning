import os
import numpy as np
import torch

class EarlyStopping:
    def __init__(self, dataset_size, optimizer, seed, lr, batchsize, depth, gsigma, patience=10):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.min_val_loss = np.Inf
        self.running_val_loss = []
        self.dataset_size = dataset_size
        self.model_name = 'DSseed%d_%s_lr%s_batchsize%d_depth%d_gsigma%d' \
                           % (seed, optimizer, 'p'.join(str(lr).split('.')), batchsize, depth, gsigma)

    def __call__(self, val_loss, model):
        self.running_val_loss.append(val_loss)
        if len(self.running_val_loss) > 11:
            self.running_val_loss = self.running_val_loss[1:]

        print('Median val loss: %.4f, min val loss: %.4f' % (np.median(self.running_val_loss), self.min_val_loss))
        # Check if there is improvement over the moving median val loss
        # But only perform a checkpoint if better than the min val loss as we want to keep the best model
        if val_loss >= np.median(self.running_val_loss):
            self.counter += 1
            print('No improvement for %d epochs' %self.counter)
            if self.counter >= self.patience:
                self.early_stop = True
        elif val_loss <= self.min_val_loss:
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.min_val_loss = val_loss
        else:
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        model_dir = os.path.join(self.model_name, 'DSsize%d' % self.dataset_size)
        model_dir = os.path.join('saved_models', model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, 'net.pth'))
        self.min_val_loss = val_loss
