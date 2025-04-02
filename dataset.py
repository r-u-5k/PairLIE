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
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        folder_path = join(self.data_dir, str(index + 1))
        data_filenames = [join(folder_path, x) for x in listdir(folder_path) if is_image_file(x)]

        chosen_files = random.sample(data_filenames, 3)

        images = []
        file_names = []
        for file in chosen_files:
            im = load_img(file)
            images.append(im)
            _, fname = os.path.split(file)
            file_names.append(fname)

        if self.transform:
            seed = np.random.randint(123456789)
            transformed_images = []
            for im in images:
                random.seed(seed)
                torch.manual_seed(seed)
                transformed_images.append(self.transform(im))
            images = transformed_images

        return images[0], images[1], images[2], file_names[0], file_names[1], file_names[2]

    def __len__(self):
        return 324


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
