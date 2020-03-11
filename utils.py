from dataloader import ImageDataset, get_loader
from settings import BATCH_SIZE, SHUFFLE


def get_datasets():
    train_dataset = get_loader(ImageDataset("trainIds.csv"), BATCH_SIZE, SHUFFLE)
    test_dataset = get_loader(ImageDataset("testIds.csv"), BATCH_SIZE, SHUFFLE)
    return train_dataset, test_dataset

