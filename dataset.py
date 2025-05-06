import os
import random
from os import listdir
from os.path import join

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.dir_m = join(data_dir, 'middle')
        self.dir_l = join(data_dir, 'low')
        self.dir_ml = join(data_dir, 'middle-low')

        self.filenames = sorted([
            fname for fname in listdir(self.dir_m)
            if is_image_file(fname)
        ])

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]

        path_m = join(self.dir_m, fname)
        path_l = join(self.dir_l, fname)
        path_ml = join(self.dir_ml, fname)

        im_m = load_img(path_m)
        im_l = load_img(path_l)
        im_ml = load_img(path_ml)

        if self.transform:
            seed = np.random.randint(123456789)
            random.seed(seed);
            torch.manual_seed(seed)
            im_m = self.transform(im_m)
            random.seed(seed);
            torch.manual_seed(seed)
            im_l = self.transform(im_l)
            random.seed(seed);
            torch.manual_seed(seed)
            im_ml = self.transform(im_ml)

        # 반환 순서는 train 루프에 맞춰 (middle, low, middle-low, fname x3)
        return im_m, im_l, im_ml, fname, fname, fname


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)
        return input, file

    def __len__(self):
        return len(self.data_filenames)
