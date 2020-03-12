import torchvision
import torch.nn as nn
import torch

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Discriminator(nn.Module):
    def __init__(self):
        """
        CNN outputs 1 if input image is real, 0 if it's fake
        """
        super(Discriminator, self).__init__()
        num_classes=1 #binary problem

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1 )
        self.batch2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1 )
        self.batch3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(25690112, num_classes)
        self.sig = nn.Sigmoid()


    def __call__(self, images):
        classify = nn.Sequential(
            self.conv1,
            self.batch1,
            self.relu,
            self.conv2,
            self.batch2,
            self.relu,
            self.conv3,
            self.batch3,
            self.relu,
            Flatten(),
            self.fc,
            self.sig
        )
        return classify(images)
'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        input_channels = 3
        n_layers = 3
        norm_layer = nn.BatchNorm2d
        use_bias = True
        ndf = 64
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_channels, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
'''