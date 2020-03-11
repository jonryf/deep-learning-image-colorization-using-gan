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



class pix2pix():

    def __init__(self, train_dataset, test_dataset):
        numclasses = 3 #RGB
        numchannels = 64
        self.gen = UNET(numclasses, numchannels)
        self.disc = Discriminator()
        self.criterion = CrossEntropyLoss()
        #self.transform = transforms.Compose([
        #    transforms.functional.to_grayscale(img, num_output_channels=1)
        #])
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


    def generate(self, greyscale):
        return self.gen.forward(greyscale)
        #Need to add dropout

    def discriminate(self, img):
        #(images, features, height, width)
        # Return average - 1 value for all images

        patch_values = self.disc.forward(img)


    def train(self, totalEpochs=EPOCHS, genLr=0.0001, descLr=0.00005):
        genOptimizer = Adam( list(self.gen.parameters()), lr=genLr)
        discOptimizer = Adam( list(self.disc.parameters()), lr=descLr)
        for epoch in totalEpochs:
            for minibatch, color_and_gray, gray_three_channel in enumerate(self.train_dataset):
                #(images, features, height, width) assume black and white images are paired in the features chanel
                origBW = color_and_gray[1]
                origColor = color_and_gray[0]

                # train descriminator
                genColor = self.generate(gray_three_channel)
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
