from dataloader import ImageDataset, get_loader
from settings import BATCH_SIZE, SHUFFLE
import pytorch_ssim
import torch
from torch.autograd import Variable

def get_datasets():
    train_dataset = get_loader(ImageDataset("trainIds.csv"), BATCH_SIZE, SHUFFLE)
    test_dataset = get_loader(ImageDataset("testIds.csv"), BATCH_SIZE, SHUFFLE)
    return train_dataset, test_dataset

def similarity(img1, img2):

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
    return pytorch_ssim.ssim(img1, img2)
