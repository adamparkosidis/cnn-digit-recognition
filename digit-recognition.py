import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# we now select the device based on what is available

if torch.cuda.is_available():                     # if a GPU is available for pytorch
  DEVICE = torch.device('cuda')
else:
  DEVICE = torch.device('cpu')


# Let's first initialize the random seeds
random_seed = 1
torch.manual_seed(random_seed)

# Then we load training data.  We use here the traditional MNIST dataset, which are 28x28 images of hand-written digits between 0 and 9

batch_size_train = 64  # by default we use a minibatch size of 64 for training.
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
          transform=torchvision.transforms.Compose([
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(
                  (0.1307,), (0.3081,)),

          ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
          transform=torchvision.transforms.Compose([
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(
                  (0.1307,), (0.3081,)),
          ])),
    batch_size=batch_size_test, shuffle=True)
