import torch
import torchvision
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from settings import EPOCHS
from unet import UNET
from Discriminator import Discriminator
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter



def trainPix2Pix(model, data, totalEpochs=EPOCHS, genLr=0.0001, descLr=0.00005):
    genOptimizer = Adam( list(model.gen.parameters()), lr=genLr)
    discOptimizer = Adam( list(model.disc.parameters()), lr=descLr)
    criterion = CrossEntropyLoss()
    model.gen.train()
    model.disc.train()
    for epoch in totalEpochs:
        for minibatch, color_and_gray, gray_three_channel in enumerate(data):
            gradientStep(model, (color_and_gray, gray_three_channel), criterion, genOptimizer, discOptimizer)


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

    def __init__(self, train_dataset, test_dataset):
        numclasses = 3 #RGB
        numchannels = 64
        self.gen = UNET(numclasses, numchannels)
        self.disc = Discriminator()
        self.criterion = CrossEntropyLoss()
        self.trainData = []
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.writer = SummaryWriter('runs/pix2pix')

    def log_image(self, images):
        # write to tensorboard
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image('four_fashion_mnist_images', img_grid)

    def log_metrics(self, epoch, loss):
        self.writer.add_scalar('training loss', loss, epoch)
        self.trainData.append(loss)

    def generate(self, greyscale):
        return self.gen.forward(greyscale)
        #Need to add dropout

    def discriminate(self, img):
        #(images, features, height, width)
        # Return average - 1 value for all images
        patch_values = self.disc.forward(img)
