import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNET
from Discriminator import Discriminator

class GAN(nn.Module):

    def __init__(self, n_class, numChan=64):
        super().__init__()
        self.generator = UNET(3,numChan=numChan)
        self.discriminator = Discriminator()

    def forward(self, x):
        generated = self.generator.forward(x)
        return self.discriminator(generated)
