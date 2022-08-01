# A convolutional network in PyTorch to recognize handwritten digits as given in the MNIST dataset

[PyTorch Documentation](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)

[Description of the dataset we will be working with](https://en.wikipedia.org/wiki/MNIST_database)


##

## Using GPU

One advantage of using pytorch as a framework is that it allows us to execute our code on the GPU. This can often greatly reduce the runtime needed to train neural networks. Below is a short description of how to do this.


## CNN

Below you can see a diagram for a convolutional network. The diagram is translated into a PyTorch model by filling in the `Net` class below.

![CNN Diagram](CNN.png)

After each max pooling step and after the first dense (linear) layer apply the relu activation function.
We use the modules `nn.Conv2d`, `nn.Linear` and the functions `F.max_pool2d` and `F.relu`.

*Hint: Carefully inspect the shapes of the intermediate layers and add padding to the convolutions where necessary.*
