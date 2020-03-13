import torch
import torchvision
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from settings import EPOCHS
from unet import UNET
from Discriminator import Discriminator
import torchvision.transforms as transforms
from torch.nn import SoftMarginLoss, BCELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import pytorch_ssim



def trainPix2Pix(model, data, totalEpochs=EPOCHS, genLr=0.0001, descLr=0.00005, useAutoencoderLoss=True):
    genOptimizer = Adam( list(model.gen.parameters()), lr=genLr)
    discOptimizer = Adam( list(model.disc.parameters()), lr=descLr)
    criterion = BCELoss()
    model.gen.train()
    model.disc.train()
    for epoch in range(totalEpochs):
        print(epoch)
        for minibatch, (color_and_gray, gray_three_channel) in enumerate(data):
            gradientStepPix2Pix(model, color_and_gray.cuda(), gray_three_channel.cuda(), criterion, genOptimizer, discOptimizer, useAutoencoderLoss=useAutoencoderLoss)

# assumes minibatch is only colord images.
def gradientStepPix2Pix(model, color, gray, criterion, genOptimizer, discOptimizer, useAutoencoderLoss=True):
    origBW = gray
    origColor = color

    # train descriminator
    genColor = model.generate(origBW)
    genPairs = torch.cat((origBW, genColor), 3)
    origPairs = torch.cat((origBW, origColor), 3)
    
    # print perf
    print(pytorch_ssim.ssim(genColor[:,:,:,:], origColor[:,:,:,:]))

    descTrainSet = torch.cat((genPairs, origPairs), 0)
    genLabels = (torch.ones(genPairs.size()[0],1).cuda()) * -1
    origLabels =  torch.ones(origPairs.size()[0],1).cuda()
    labels = torch.cat((genLabels,origLabels),0)
    preds = model.discriminate(descTrainSet)
#     print(("labels: ", labels))
    print(("predictions", preds))
    batch_loss = criterion(preds, labels)
#     print(("batch_loss: ",batch_loss))

    model.gen.zero_grad()
    model.disc.zero_grad()
    batch_loss.backward()
    discOptimizer.step()

    genColor = model.generate(origBW)
    genPairs = torch.cat((origBW, genColor), 3)
    preds = model.discriminate(genPairs)
    batch_loss = criterion(preds,  torch.ones(genLabels.size()[0], 1).cuda())
    print(("batchLoss",batch_loss))
    
    if useAutoencoderLoss:
        diff = genColor - origColor
        diff = diff * diff
        diff = torch.sum(diff)
        batch_loss += diff
    
    model.gen.zero_grad()
    model.disc.zero_grad()
    batch_loss.backward()
    genOptimizer.step()


class pix2pix(nn.Module):

    def __init__(self):
        super(pix2pix, self).__init__()
        numclasses = 3 #RGB
        numchannels = 64
        self.gen = UNET(numclasses, numchannels)
        self.disc = Discriminator()
#         self.criterion = CrossEntropyLoss()
        self.writer = SummaryWriter('runs/pix2pix')

    def log_image(self, images):
        # write to tensorboard
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image('four_fashion_mnist_images', img_grid)

    def log_metrics(self, epoch, loss):
        self.writer.add_scalar('training loss', loss, epoch)
        self.trainData.append(loss)

    def generate(self, greyscale):
        return self.gen(greyscale)
        #Need to add dropout

    def discriminate(self, img):
        #(images, features, height, width)
        # Return average - 1 value for all images
        ret = self.disc(img)
        ret = torch.mean(ret, axis=2)
        ret = torch.mean(ret, axis=2)
        return ret
