from pix2pix import pix2pix
from utils import get_datasets


def run_network():
    train_dataset, test_dataset = get_datasets()
    network = pix2pix(train_dataset, test_dataset)

    network.train()


run_network()

