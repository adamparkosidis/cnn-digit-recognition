import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding='same')
    self.linear1 = nn.Linear(720,50)
    self.linear2 = nn.Linear(50,10)

  def forward(self, x):
    x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
    x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))

    batch_size = x.shape[0]
    x2 = x2.view(batch_size,-1)

    y1 = torch.nn.functional.relu(self.linear1(x2))
    y = self.linear2(y1)
    return y

# Now we define the optimizer, and instantiate the network.

learning_rate = 0.01

network = Net().to(DEVICE)  # We move the network to the GPU
optimizer = optim.Adam(network.parameters(), lr=learning_rate)


n_epochs = 3  # 3 epochs by default

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

# This is the main training loop

log_interval = 10

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    data = data.to(DEVICE)
    target = target.to(DEVICE)
    output = network(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

# This is the main testing loop

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data = data.to(DEVICE)
      target = target.to(DEVICE)
      output = network(data)
      test_loss += F.cross_entropy(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Let's train our model

test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')


with torch.no_grad():
    output = network(example_data.to(DEVICE))

fig = plt.figure(figsize=(20, 10))
for i in range(40):
  plt.subplot(5,8,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
      output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
