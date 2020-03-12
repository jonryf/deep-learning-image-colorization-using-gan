import random

from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import math
import pandas as pd
from collections import namedtuple
import torchvision.transforms.functional as TF
import torchvision

from settings import NUM_WORKERS


class ImageDataset(Dataset):

    def __init__(self, csv_file):
        #self.data = pd.read_csv(csv_file, header=None)
        #csv = pd.read_csv(csv_file, header=None)
        self.data = []
        with open(csv_file) as f:
            self.data = f.readlines()
        self.data = [x.strip() for x in self.data]
        # Add any transformations here

    def __len__(self):
        return len(self.data)

    @staticmethod
    def transform_data(image):
        h,w = image.size
        size = min(h,w)
        transform = transforms.Compose([
            torchvision.transforms.CenterCrop(size),
            torchvision.transforms.Resize(224)])
        image = transform(image)
        return image

    def __getitem__(self, idx):
        img_name = './images/{}.jpeg'.format(self.data[idx])

        img = Image.open(img_name).convert('RGB')

        gray_scale = img.convert('L')

        # apply transformation
        img = self.transform_data(img)

        gray_scale = self.transform_data(gray_scale)

        

        gray_scale = np.asarray(gray_scale)

        gray_three_channel = np.dstack((gray_scale, gray_scale, gray_scale))

        img = np.asarray(img)

        color_and_gray = np.concatenate((gray_three_channel, img), axis=0)
        #color_and_gray = np.concatenate((gray_three_channel, img), axis=3)

        # convert to tensor
        color_and_gray = torch.from_numpy(color_and_gray.copy()).float()
        gray_three_channel = torch.from_numpy(gray_three_channel.copy()).float()

        return color_and_gray, gray_three_channel


def get_loader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=NUM_WORKERS)


