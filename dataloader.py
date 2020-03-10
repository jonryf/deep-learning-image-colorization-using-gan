import random

from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import math
import pandas as pd
from collections import namedtuple
import torchvision.transforms.functional as TF


class ImageDataset(Dataset):

    def __init__(self, csv_file, transform, n_class=n_class):
        self.data      = pd.read_csv(csv_file)
        # Add any transformations here
        self.transform = transforms.Compose([
            transforms.Resize(254),
            transforms.RandomCrop((254, 254), pad_if_needed=True),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def transform_data(image):

        return image


    def __getitem__(self, idx):
        img_name   = './images/{}.jpeg'.format(self.data.columns[idx])

        img = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # apply transformation
        img = self.transform_data(img)

        gray_scale = img.convert('L')

        gray_scale = np.asarray(gray_scale)

        gray_three_channel = np.dstack((gray_scale, gray_scale, gray_scale))

        img = np.asarray(img)

        color_and_gray = np.concatenate((gray_three_channel, img), axis=0)


        # convert to tensor
        color_and_gray = torch.from_numpy(color_and_gray.copy()).float()
        gray_three_channel = torch.from_numpy(gray_three_channel.copy()).float()





        return color_and_gray, gray_three_channel