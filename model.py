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


