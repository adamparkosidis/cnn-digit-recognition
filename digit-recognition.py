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

# To train any Pytorch ML model, we need four things: a model definition,
# a criterion (a loss function), an optimizer (gradient descent), and a training routine.

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
# ```torchvision.transforms.ToTensor()``` and also normilizes the data ```torchvision.transforms.Normalize((0.1307,), (0.3081,))```,
# where the parameters are the mean and standard deviation of the MNIST dataset. To tackle the model learning difficulty, we
# normalize training data. We ensure that the value ranges of the various features are comparable (feature scaling) so that gradient
# descents can converge faster. Finally, ```shuffle=True```, because it aids in the training's rapid convergence, it eliminates any
# prejudice during training and inhibits the model from learning the training order.

examples = list(test_loader)
example_data, example_targets = examples[0]
print(example_data.shape)
print(example_targets.shape)

# The `example_data` contains 1000 object ```len(example_data)```=1000, where each element is an image of hand-written digits between
# 0 and 9. Each of these objects contains 28 arrays ```len(example_data[0][0])=28``` and each of the 28 arrays contains 28 elements
#```len(example_data[0][0][0])=28```. These are the 28x28 pixels of each image (28 rows x 28 columns). Each of these
# 784 elements contains a value between [-0.4242, 2.8215] which indicates the color of the pixel, where -0.4242 corresponds to total
# black while 2.8215 to total white. The printed shape of the ```example_data``` is ```torch.Size([1000, 1, 28, 28])```, where number 1 just
# states that the array has 1000 rows and 1 column. This is because ```example_data``` is a tensor and we need it in this form to able to
# perform tensor multiplications. The ```example_targets``` tensor contains 1000 elements, where each elements represent the actual digit
# that corresponts to each image of the ```example_data```. For example, ```example_targets[0] = 3``` which means that ```example_data[0]``` containts
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

# The next step is to define a convolutional neural network.
  
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

# Now we define the optimizer, and initiate the network.

learning_rate = 0.01

network = Net().to(DEVICE)  # We move the network to the GPU
optimizer = optim.Adam(network.parameters(), lr=learning_rate)  # Here we are using the so called [Adam](https://arxiv.org/abs/1412.6980) optimizer. 

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

# Let's train our model and print some results

test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# The model has an accuracy around 98%!
  
# We can visualize how the loss function is minimizing 
    
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss') 

# It is also nice to visualize some of the test data with the model's prediction

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
  
########################################################################################################## 
  
#### BONUS ###  

# As bonus we will compare our model with a one-layer perceptron (i.e. no hidden layers). We will write a new `class SimplePerceptron` that flattens
# the 28 x 28 images passes them through one fully connected linear layer with input size 28 times 28 and outputs a 10 dimensional one-hot vector. 
# We will see the classification accuracy in this case and we will compare it to the accuracy from the CNN from before.

class SimplePerceptron(torch.nn.Module):
  def __init__(self):
      super(SimplePerceptron, self).__init__()
      self.linear = torch.nn.Linear(784,10) 
      pass 

  def forward(self, x):
    batch_size = x.shape[0]
     
    x = x.view(batch_size,-1)
    y = self.linear(x)
    return y


learning_rate = 0.01  # same learning rate as before

network =  SimplePerceptron().to(DEVICE)  # We move the network to the GPU
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

n_epochs = 3  # 3 epochs by default

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

# Let's train our model and print some results

test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
    
# The classification accuracy in this case is around 89%. It maybe be surprising that such a simple model can actually provide us with 
# such an accuracy based on the fact that it consists only by a linear transformation. The accuracy of the CNN is approximatelly 10% higher. 
# This is of course a big improvement.

# Let's make the fitting a bit more challenging. We now rotate every image by [0,180] degrees randomly.

batch_size_train = 64  # by default we use a minibatch size of 64 for training.

batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
          transform=torchvision.transforms.Compose([
              torchvision.transforms.RandomRotation(180),                          # random rotation
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize( 
                  (0.1307,), (0.3081,)),
                             
          ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
          transform=torchvision.transforms.Compose([
              torchvision.transforms.RandomRotation(180),                          # random rotation
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(
                  (0.1307,), (0.3081,)),
          ])),
    batch_size=batch_size_test, shuffle=True)

examples = list(test_loader)
example_data, example_targets = examples[0]
print(example_data.shape)
print(example_targets.shape)

# Let's visualize some of our test data.

fig = plt.figure(figsize=(20, 10))
for i in range(40):
  plt.subplot(5,8,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
  
# Let's test the one-layer perceptron (i.e. no hidden layers)
  
learning_rate = 0.01

network =  SimplePerceptron2().to(DEVICE)  # We move the network to the GPU
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

n_epochs = 3  # 3 epochs by default.  Leave it like that throughout the subsequent exercises.

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# Let's test the CNN  
  
learning_rate = 0.01

network = Net().to(DEVICE)  # We move the network to the GPU
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

n_epochs = 3  # 3 epochs by default

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

# Let's train our model and print some results

test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
    
# We can clearly see that the CNN performs pretty well and most importantly much better than the simple model. The reason behind that is that the CNN is
# able to bring the data in a well feature-organized form making it much easier for the model to train on specific features. More specifically, with every
# convolution the data is categorized in a number of channels where each channel preserves specific features of the initial image, then the max-pooling 
# operation compresses the data so that the model can learn faster, while preserves the info structure of the initial image. By repeating the 
# aforementioned process (convolution+max-pooling) the model is able to repeat the feature categorization but for more complex features. At the end of
# the repeating convolutions and maxi-pooling operations the data is delivered in a form that the model is able to be trained in specific features 
# (e.g. circles, lines etc) and in to correlations between them. In other words, the CNN is able to predict the numbers even if we disrupt the global 
# structure of the data (rotation), because the network is categorizing different features of the data and the model trained in these and the correlations
# between them. On the other hand, a simple linear layer model is very dependent on the global structure of the image, because the model is not trained
# on many different specific features and thus it performs poorly after rotating the data (disrupting the global info structure)
