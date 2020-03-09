import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class pix2pix():

    def __init__(self):
        #encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        #decoder
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=0, output_padding=0)
    def generate(self):
        #Encoder-Decoder CNN-TransposeCNN

        pass

    def discriminate(self):
        #CNN-Sigmoid
        pass

    def train(self, minibatch):
        fake_label = 0
        real_label = 1
        #generate
        #backprop D
        #discriminate
        #backprop G
