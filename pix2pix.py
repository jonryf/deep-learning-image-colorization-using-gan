import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from unet import UNET
from Discriminator import Discriminator
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def trainPix2Pix(model, data, batchSize=1 ,totalEpochs=50, genLr=0.0001, descLr=0.00005):
    genOptimizer = Adam( list(model.gen.parameters()), lr=genLr)
    discOptimizer = Adam( list(model.disc.parameters()), lr=descLr)
    criterion = CrossEntropyLoss()
    model.gen.train()
    model.disc.train()
    for epoch in totalEpochs:
        for index in range(0, data.size()[0], batchSize):
            curData = data[index:index + batchSize,:,:,:]
            gradientStep(model, genInputs, discImputs, genOptimizer, discOptimizer)


# assumes minibatch is only colord images.
def gradientStepPix2Pix(model, minibatch, criterion, genOptimizer, discOptimizer):
    origBW = blackandWhite(minibatch)
    origColor = minibatch

    # train descriminator
    genColor = model.generate(origBW)
    genPairs = torch.cat((origBW, genColor), 3)
    origPairs = torch.cat((origBW, origColor), 3)

    descTrainSet = torch.cat((genPairs, origPairs), 0)
    genLabels = torch.zeros(genPairs.size()[0])
    origLabels =  torch.ones(origPairs.size()[0])
    labels = torch.cat(genLabels,origLabels),0))
    preds = model.discriminate(descTrainSet)
    batch_loss = criterion(preds, labels)

    model.gen.zero_grad()
    model.disc.zero_grad()
    batch_loss.backward()
    discOptimizer.step()

    preds = model.discriminate(genPairs)
    batch_loss = criterion(preds,  torch.ones(genLabels.size()[0]))
    model.gen.zero_grad()
    model.disc.zero_grad()
    batch_loss.backward()
    genOptimizer.step()


class pix2pix():

    def __init__(self):
        numclasses = 3 #RGB
        numchannels = 64
        self.gen = UNET(numclasses, numchannels)
        self.disc = Discriminator()
        self.criterion = CrossEntropyLoss()

    def generate(self, greyscale):
        return self.gen.forward(greyscale)
        #Need to add dropout

    def discriminate(self, img):
        #(images, features, height, width)
        # Return average - 1 value for all images
        patch_values = self.disc.forward(img)