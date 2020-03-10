import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from unet import UNET
from Discriminator import Discriminator

class pix2pix():

    def __init__(self):
        numclasses = 3 #RGB
        numchannels = 64
        self.gen = UNET(numclasses, numchannels)
        self.disc = Discriminator()

    def generate(self, greyscale):
        return self.gen.forward(greyscale)
        #Need to add dropout

    def discriminate(self, img):
        #(images, features, height, width)
        # Return average - 1 value for all images

        patch_values = self.disc.forward(img)

        pass

    def train(self, all_batches):
        fake_label = 0
        real_label = 1
        for minibatch in all_batches:
            #make BW versions
            #generate coloration
            generations = self.generate(____)
            #concatenate generations w BW images
            #concatenate ground truth w BW images
            #create training batch for D: (BW-color concatenated images, both real and fake)
            #run training data through D
            #get predictions
            #compare with ground truth labels
            #backprop D
            #pass generated examples back into D
            #get predictions
            #compare with ground truth labels (all 0's)
            #backprop G

        #discriminate
        #backprop G
