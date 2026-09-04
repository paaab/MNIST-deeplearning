import torch.nn as nn

class SimpleNN(nn.Module):
    #inherits from nn.Module, a PyTorch class
    #it provides methods to manage layers and weights and save/load models

    def __init__(self):
        #init is the constructor of the class, executed automatically when we call it
        #self references the current object
        super().__init__() #we use super() so that when we use the constructor, the constructor of the nn.Module class is executed

        self.fc1 = nn.Linear(784, 128)
        #This is our first layer, named fc1 (fully connected, layer1), it receives 784 pixels and produces 128 values, one from each neuron
        #a neuron is a small mathematical unit which combines pixels with weights
        #f.e. 3 pixels: [2, 5, 3], with (firstly random) 3 weights: [0.2, -0.4, 0.7] -> 2x0.2 + 5x-0.4 + 3x0.7 = 0.5 is the final output for this neuron

        self.relu = nn.ReLU()
        #if we only had lineal layers, the model would not be able to learn more complex relationships, thats why we add ReLU (Rectified Linear Unit)
        #it is quite simple, it just converts negatives values from the outputs and turn them into 0 (ReLU(x) = max(0, x))
        #this ads non-linearity to our model

        self.fc2 = nn.Linear(128, 10)
        #same idea as fc1, but why 10? MNIST has 10 possible classes
        #this 10 values are named logits, they are scores that show how much the model favors a class, not probabilities yet

    def forward(self, x):
        #This is the "path" that our data is going to follow, x is what we called the images that our model is going to receive

        x = x.view(x.size(0), -1)
        #we reorganize data, by now we have our batch has [32 (batch size), 1 (channel), 28, 28 (size of the image, 28x28)]
        #but our first layer needs 784 pixels, so we use view: x.size(0) returns the first value of x (32), and -1 calculates the remaining data (1x28x28)
        #now we have our data flattened: [32, 784]

        x = self.fc1(x) #data goes through fc1, 784 pixels -> 128 values

        x = self.relu(x) #we apply ReLU, here the shape doesnt change, 128 -> 128, but we "turn off" negative values

        x = self.fc2(x) #applies fc2

        return x #returns final output



