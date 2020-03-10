import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from unet import UNET
from Discriminator import Discriminator
import torchvision.transforms as transforms

class pix2pix(nn.Module):

    def __init__(self):
        numclasses = 3 #RGB
        numchannels = 64
        self.gen = UNET(numclasses, numchannels)
        self.disc = Discriminator()
        self.transform = transforms.Compose([
            transforms.Resize(254),
            transforms.RandomCrop((254, 254), pad_if_needed=True),
            transforms.ToTensor()])

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
            BW = torch.transforms.Greyscale(out=3)
            #generate coloration
            generations =self.generate(BW)
            #concatenate generations w BW images
            fakes = torch.cat((BW, generations), 3)
            #concatenate ground truth w BW images
            reals = torch.cat((BW, minibatch), 3)
            #create training batch for D: (BW-color concatenated images, both real and fake)
            trainBatch = torch.cat(fakes, reals)
            #run training data through D
            #get predictions
            pred = self.discriminate(trainBatch)
            #compare with ground truth labels
            zerolabels = torch.zeros(1, minibatch.shape[0])
            onelabels = torch.ones(1, minibatch.shape[0])
            #backprop D

            #pass generated examples back into D
            #get predictions
            #compare with ground truth labels (all 0's)
            #backprop G

        #discriminate
        #backprop G
