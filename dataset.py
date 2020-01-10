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
    '''
    Divide the image into smaller patches
    '''

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
    '''
    Create the h5py dataset from the 400 base images in the BSD400 dataset
    Inputs:
        data_path: the folder where the BSD400 dataset is stored
        patch_size: the size of the smaller patches
        stride: the stride used to create the patches
        gksize: the size of the gaussian blur kernel
        gsigma: the sigma parameter of the gaussian blur kernel
    '''

    print('processing data')

    # Create the gaussian blur kernel
    kernel = Kernels.kernel_2d(gksize, gsigma)

    files = glob.glob(os.path.join(data_path, 'BSD400', '*.png'))
    files.sort()
    train_file_name = 'datasets/train_ps%d_stride%d.h5' % (patch_size, stride)

    # Create the h5py file
    h5f = h5py.File(train_file_name, 'w')
    train_num = 0
    for i in range(len(files)):
        # Read the base image
        Img = cv2.imread(files[i])

        h, w, c = Img.shape

        Img = np.expand_dims(Img[:,:,0].copy(), 0)
        Img = np.float32(normalize(Img))

        # Divide the image into smaller patches
        patches = Im2Patch(Img, win=patch_size, stride=stride)
        print("file: %s # samples: %d" % (files[i], patches.shape[3]), end='\r')
        for n in range(patches.shape[3]):
            patch = np.expand_dims(patches[:,:,:,n].copy(), -1)
            # Add the blurred dimension as a new entry on the last dimension
            blurred = np.expand_dims(
                np.expand_dims(
                    cv2.filter2D(patch[0], -1, kernel, borderType=cv2.BORDER_CONSTANT),
                    -1
                ),
                0
            )
            data = np.concatenate((patch, blurred), axis=-1)
            h5f.create_dataset(str(train_num), data=data)
            train_num += 1
    h5f.close()

    print('\ntraining set, # samples %d\n' % train_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True, patch_size=64, stride=32, size='full', seed=1234, mode=None):
        super(Dataset, self).__init__()
        self.train = train

        # Store the h5py filename
        if mode == 'augmented':
            self.train_file_name = 'datasets/train_ps%d_stride%d_augmented.h5' % (patch_size, stride)
        elif mode == 'vae':
            self.train_file_name = 'datasets/train_ps%d_stride%d_vae.h5' % (patch_size, stride)
        else:
            self.train_file_name = 'datasets/train_ps%d_stride%d.h5' % (patch_size, stride)
        h5f = h5py.File(self.train_file_name, 'r')

        self.all_keys = sorted(list(h5f.keys()))
        # shuffle the keys according to the seed
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


def augment_dataset(patch_size=64, stride=32, size='full', seed=1234):
    '''
    Augment a dataset with traditionnal methods
    Inputs:
        patch_size: the patch_size of the dataset to augment
        stride: the stride of the dataset to augment
        size: the size of the dataset to use for augmentation
        seed: the seed of the dataset to use for augmentation
    '''

    print('augmenting dataset')

    # Load the dataset
    ds = Dataset(patch_size=patch_size, stride=stride, size=size, seed=seed)
    loader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle=False
    )

    # Create the new h5py file for the augmented dataset
    augmented_file_name = 'datasets/train_ps%d_stride%d_augmented.h5' % (patch_size, stride)
    h5f = h5py.File(augmented_file_name, 'w')
    # Each image can have 4 different rotations
    num_rots = 4

    for patch_num, patch in enumerate(loader):
        patch = patch[0].numpy()
        # Rotations of the image
        for i in range(num_rots):
            new_patch = np.expand_dims(np.rot90(patch[0], i), axis=0)
            h5f.create_dataset(str(2 * num_rots * patch_num + i), data=new_patch)

        # Rotations of the mirrored image
        patch = patch[:,::-1]
        for i in range(num_rots):
            new_patch = np.expand_dims(np.rot90(patch[0], i), axis=0)
            h5f.create_dataset(str(2 * num_rots * patch_num + i + num_rots), data=new_patch)

    h5f.close()
    print('dataset augmentation done')
