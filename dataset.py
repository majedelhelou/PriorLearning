import os
import numpy as np
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from util import Kernels


def normalize(data):
    return data / 255.


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]

    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):

            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data(data_path='./data', patch_size=64, stride=32, gksize=11, gsigma=3):
    print('processing data')

    kernel = Kernels.kernel_2d(gksize, gsigma)

    files = glob.glob(os.path.join(data_path, 'BSD400', '*.png'))
    files.sort()
    train_file_name = 'datasets/train_ps%d_stride%d.h5' % (patch_size, stride)

    h5f = h5py.File(train_file_name, 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])

        Img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        h, w, c = Img.shape

        Img = np.expand_dims(Img[:,:,0].copy(), 0)
        Img = np.float32(normalize(Img))

        patches = Im2Patch(Img, win=patch_size, stride=stride)
        print("file: %s # samples: %d" % (files[i], patches.shape[3]), end='\r')
        for n in range(patches.shape[3]):
            data = patches[:,:,:,n].copy()
            h5f.create_dataset(str(train_num), data=data)
            train_num += 1
    h5f.close()

    print('\ntraining set, # samples %d\n' % train_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True, patch_size=32, stride=16, size='full', seed=1234):
        super(Dataset, self).__init__()
        self.train = train

        self.train_file_name = 'datasets/train_ps%d_stride%d.h5' % (patch_size, stride)
        h5f = h5py.File(self.train_file_name, 'r')

        self.all_keys = sorted(list(h5f.keys()))
        np.random.seed(seed)
        np.random.shuffle(self.all_keys)

        self.size = len(self.all_keys) if size == 'full' else size
        self.keys = self.all_keys[:self.size]
        h5f.close()

    def set_size(self, size='full'):
        self.size = len(self.all_keys) if size == 'full' else size
        self.keys = self.all_keys[:self.size]

    def is_full(self):
        return self.size == len(self.all_keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.train_file_name, 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
