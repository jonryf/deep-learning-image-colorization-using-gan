import torch
import torch.nn as nn
import torch.nn.functional as F

# gets centered crop of the fiven tensor
# image dimmensions = 1024 x 2048
# torch.Size([1, 1, 32, 32])
# size (images, features, height, width)
def tensorCenterCrop(tensor, height, width):
    heightStartIdx = ((tensor.size()[2] +1) - height) / 2
    widthStartIdx = ((tensor.size()[3] +1) - width) / 2
    return tensor[:,:,int(heightStartIdx):int(heightStartIdx+height), int(widthStartIdx):int(widthStartIdx+width)]



# torch.cat((first_tensor, second_tensor), 0)

class UNET(nn.Module):

    def __init__(self, n_class, numChan=64):
        super().__init__()
        self.n_class = n_class

        self.conv1_1   = nn.Conv2d(3, numChan, kernel_size=3, stride=1, padding=1)
        self.conv1_2   = nn.Conv2d(numChan, numChan, kernel_size=3, stride=1, padding=1 )
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=False, ceil_mode=False)

        self.conv2_1   = nn.Conv2d(numChan, (numChan * 2), kernel_size=3, stride=1, padding=1 )
        self.conv2_2   = nn.Conv2d((numChan * 2), (numChan * 2), kernel_size=3, stride=1, padding=1 )
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=False, ceil_mode=False)

        self.conv3_1   = nn.Conv2d((numChan * 2), (numChan * 4), kernel_size=3, stride=1, padding=1 )
        self.conv3_2   = nn.Conv2d((numChan * 4), (numChan * 4), kernel_size=3, stride=1, padding=1 )
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=False, ceil_mode=False)

        self.conv4_1   = nn.Conv2d((numChan * 4), (numChan * 8), kernel_size=3, stride=1, padding=1 )
        self.conv4_2   = nn.Conv2d((numChan * 8), (numChan * 8), kernel_size=3, stride=1, padding=1 )
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=False, ceil_mode=False)

        self.conv5_1   = nn.Conv2d((numChan * 8), (numChan * 16), kernel_size=3, stride=1, padding=1 )
        self.conv5_2   = nn.Conv2d((numChan * 16), (numChan * 16), kernel_size=3, stride=1, padding=1 )
        self.deconv5 = nn.ConvTranspose2d((numChan * 16), (numChan * 8), kernel_size=2, stride=2, padding=0, output_padding=0)

        self.conv6_1   = nn.Conv2d((numChan * 16), (numChan * 8), kernel_size=3, stride=1, padding=1 )
        self.conv6_2   = nn.Conv2d((numChan * 8), (numChan * 8), kernel_size=3, stride=1, padding=1 )
        self.deconv6 = nn.ConvTranspose2d((numChan * 8), (numChan * 4), kernel_size=2, stride=2, padding=0, output_padding=0)

        self.conv7_1   = nn.Conv2d((numChan * 8), (numChan * 4), kernel_size=3, stride=1, padding=1 )
        self.conv7_2   = nn.Conv2d((numChan * 4), (numChan * 4), kernel_size=3, stride=1, padding=1 )
        self.deconv7 = nn.ConvTranspose2d((numChan * 4), (numChan * 2), kernel_size=2, stride=2, padding=0, output_padding=0)

        self.conv8_1   = nn.Conv2d((numChan * 4), (numChan * 2), kernel_size=3, stride=1, padding=1 )
        self.conv8_2   = nn.Conv2d((numChan * 2), (numChan * 2), kernel_size=3, stride=1, padding=1 )
        self.deconv8 = nn.ConvTranspose2d((numChan * 2), numChan, kernel_size=2, stride=2, padding=0, output_padding=0)

        self.conv9_1   = nn.Conv2d((numChan * 2), numChan, kernel_size=3, stride=1, padding=1 )
        self.conv9_2   = nn.Conv2d(numChan, numChan, kernel_size=3, stride=1, padding=1 )
        
        
        self.classifier = nn.Conv2d(numChan, self.n_class, kernel_size=1, stride=1, padding=0, )

    def forward(self, x):
        torch.cuda.empty_cache()

        outConv1 = F.relu(self.conv1_1(x))
#         outConv1 = F.relu(self.conv1_2(outConv1))
        out1 = self.pool1(outConv1)

        outConv2 = F.relu(self.conv2_1(out1))
#         outConv2 = F.relu(self.conv2_2(outConv2))
        out2 = self.pool2(outConv2)

        outConv3 = F.relu(self.conv3_1(out2))
#         outConv3 = F.relu(self.conv3_2(outConv3))
        out3 = self.pool3(outConv3)
        
        outConv4 = F.relu(self.conv4_1(out3))
#         outConv4 = F.relu(self.conv4_2(outConv4))
        out4 = self.pool4(outConv4)

        outConv5 = F.relu(self.conv5_1(out4))
#         outConv5 = F.relu(self.conv5_2(outConv5))
        out5 = self.deconv5(outConv5)

        outConv6 = F.relu(self.conv6_1(torch.cat((out5, tensorCenterCrop(outConv4, out5.size()[2], out5.size(3))), 1)))
#         outConv6 = F.relu(self.conv6_2(outConv6))
        out6 = self.deconv6(outConv6)

        outConv7 = F.relu(self.conv7_1(torch.cat((out6, tensorCenterCrop(outConv3, out6.size()[2], out6.size(3))), 1)))
#         outConv7 = F.relu(self.conv7_2(outConv7))
        out7 = self.deconv7(outConv7)

        outConv8 = F.relu(self.conv8_1(torch.cat((out7, tensorCenterCrop(outConv2, out7.size()[2], out7.size(3))), 1)))
#         outConv8 = F.relu(self.conv8_2(outConv8))
        out8 = self.deconv8(outConv8)

        outConv9 = F.relu(self.conv9_1(torch.cat((out8, tensorCenterCrop(outConv1, out8.size()[2], out8.size(3))), 1)))
#         outConv9 = F.relu(self.conv9_2(outConv9))
        
        
        preds = self.classifier(outConv9)
        torch.cuda.empty_cache()

        return preds  # size=(N, n_class, x.H/1, x.W/1)
