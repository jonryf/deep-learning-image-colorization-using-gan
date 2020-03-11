import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from unet import UNET
from Discriminator import Discriminator
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class pix2pix():

    def __init__(self):
        numclasses = 3 #RGB
        numchannels = 64
        self.gen = UNET(numclasses, numchannels)
        self.disc = Discriminator()
        self.criterion = CrossEntropyLoss()
        self.transform = transforms.Compose([
            transforms.functional.to_grayscale(img, num_output_channels=1)
        ])
        self.trainData = []
        self.


    def generate(self, greyscale):
        return self.gen.forward(greyscale)
        #Need to add dropout

    def discriminate(self, img):
        #(images, features, height, width)
        # Return average - 1 value for all images

        patch_values = self.disc.forward(img)

        pass

    def train(self, all_batches, totalEpochs=50, genLr=0.0001, descLr=0.00005):
        genOptimizer = Adam( list(self.gen.parameters()), lr=genLr)
        discOptimizer = Adam( list(self.disc.parameters()), lr=descLr)
        for epoch in totalEpochs:
            for minibatch in all_batches:
                #(images, features, height, width) assume black and white images are paired in the features chanel
                origBW = blackandWhite(minibatch)
                origColor = minibatch

                # train descriminator
                genColor = self.generate(origBW)
                genPairs = torch.cat((origBW, genColor), 3)
                origPairs = torch.cat((origBW, origColor), 3)

                descTrainSet = torch.cat((genPairs, origPairs), 0)
                genLabels = torch.zeros(genPairs.size()[0])
                origLabels =  torch.ones(origPairs.size()[0])
                labels = torch.cat(genLabels,origLabels),0))
                preds = self.discriminate(descTrainSet)
                batch_loss = self.criterion(preds, labels)

                self.gen.zero_grad()
                self.disc.zero_grad()
                batch_loss.backward()
                discOptimizer.step()

                preds = self.discriminate(genPairs)
                batch_loss = self.criterion(preds,  torch.ones(genLabels.size()[0]))
                self.gen.zero_grad()
                self.disc.zero_grad()
                batch_loss.backward()
                genOptimizer.step()