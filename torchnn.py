# Import the necessary libraries
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get the data
train = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
dataset = DataLoader(train, 32)
# 1*28*28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                                    nn.Conv2d(1, 32, (3, 3)),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, (3, 3)),
                                    nn.ReLU(),
                                    nn.Conv2d((64, 64, (3, 3))),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(64*(28-6)*(28-6), 10)
                                    )


    def forward(self, x):
        return self.model(x)