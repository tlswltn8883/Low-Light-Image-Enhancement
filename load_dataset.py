from torch.utils.data import Dataset
import tqdm
import random
import numpy as np
import torch
from PIL import Image


def image_read(short_expo_files, long_expo_files):
    short_list = []
    long_list = []

    for i in tqdm.tqdm(range(len(short_expo_files))):
        img_short = np.array(Image.open(short_expo_files[i]).convert("RGB"))
        short_list.append(img_short)

        img_long = np.array(Image.open(long_expo_files[i]).convert("RGB"))
        long_list.append(img_long)

    return short_list, long_list


def data_augmentation(image, mode):
    if mode == 0:
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image)
    elif mode == 3:
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        out = np.rot90(image, k=2)
    elif mode == 5:
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        out = np.rot90(image, k=3)
    elif mode == 7:
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out


class load_data(Dataset):
    def __init__(self, short_expo_files, long_expo_files, patch_size=128, training=True):
        self.training = training
        self.patch_size = patch_size

        if self.training:
            self.short_list, self.long_list = image_read(short_expo_files, long_expo_files)
            print('Train files loaded ......')
        else:
            self.short_list, self.long_list = image_read(short_expo_files, long_expo_files)
            print('Test files loaded ......')

    def __len__(self):
        return len(self.short_list)

    def __getitem__(self, idx):
        img_short = self.short_list[idx]
        img_long = self.long_list[idx]

        H, W, _ = img_short.shape

        if self.training:
            i = random.randint(0, H - self.patch_size)
            j = random.randint(0, W - self.patch_size)
            img_short_crop = img_short[i:i + self.patch_size, j:j + self.patch_size, :]
            img_long_crop = img_long[i:i + self.patch_size, j:j + self.patch_size, :]
            img_long_crop, img_short_crop = random_augmentation(img_long_crop, img_short_crop)

        else:
            img_short_crop = img_short
            img_long_crop = img_long

        img_short_crop = img_short_crop.astype(np.float32) / 255.0
        img_long_crop = img_long_crop.astype(np.float32) / 255.0

        img_short = torch.from_numpy(np.transpose(img_short_crop, [2, 0, 1])).float()
        img_long = torch.from_numpy(np.transpose(img_long_crop, [2, 0, 1])).float()

        return img_short, img_long