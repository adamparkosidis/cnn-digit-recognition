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

# The transform statement is used to transform the data. More specifically, we see that transforms the data in to tensor type
# (`torchvision.transforms.ToTensor()`) and also normilizes the data (`torchvision.transforms.Normalize(0.1307,), (0.3081,)`),
# where the parameters are the mean and standard deviation of the MNIST dataset. To tackle the model learning difficulty, we
# normalize training data. We ensure that the value ranges of the various features are comparable (feature scaling) so that gradient
# descents can converge faster. Finally, `shuffle=True`, because it aids in the training's rapid convergence, it eliminates any
# prejudice during training and inhibits the model from learning the training order.

examples = list(test_loader)
example_data, example_targets = examples[0]
print(example_data.shape)
print(example_targets.shape)

# The `example_data` contains 1000 object (`len(example_data)`)=1000, where each element is an image of hand-written digits between
# 0 and 9. Each of these objects contains 28 arrays (`len(example_data[0][0])`=28) and each of the 28 arrays contains 28 elements
#(`len(example_data[0][0][0])`=28). These are the $28 \times 28$ pixels of each image (28 rows $\times$ 28 columns). Each of these
# 784 elements contains a value between [-0.4242, 2.8215] which indicates the color of the pixel, where $-0.4242$ corresponds to total
# black while $2.8215$ to total white. The printed shape of the `example_data` is `torch.Size([1000, 1, 28, 28])`, where number 1 just
# states that the array has 1000 rows and 1 column. This is because `example_data` is a tensor and we need it in this form to able to
# perform tensor multiplications. The `example_targets` tensor contains 1000 elements, where each elements represent the actual digit
# that corresponts to each image of the `example_data`. For example, `example_targets[0] = 3` which means that `example_data[0]` containts
# an image of the hand-written number 3.


# Let's visualize some of our test data.

fig = plt.figure(figsize=(20, 10))
for i in range(40):
  plt.subplot(5,8,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
