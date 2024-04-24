# Import the necessary libraries
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get the data
train = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
dataset = DataLoader(train, 32)
