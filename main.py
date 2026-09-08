from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import SimpleNN

transform = transforms.ToTensor() #neural networks work with tensors instead of images, transforms images into tensors,
                                  #dividing each pixel by 255 in order to get values between 0.0 and 1.0

train_dataset = datasets.MNIST(
    root="data", #saves dataset in "data"
    train=True, #MNIST is divided in 2 sets, train=True takes the train set (60,000 samples)
    download=True, #downloads if needed, only the first time the project is executed
    transform=transform #uses the transform defined previously for each image
)

train_loader = DataLoader(
    train_dataset, #we're working with this dataset

    batch_size=32,
    #neural networks are designed to work with sets of samples, named batches
    #this is because modern CPUs are able to proces sets of images efficiently in parallel
    #32 is not a magical number, its called a hyperparameter
    #it could have been any other like 16, 64, even 128, but 32 is frequently balanced

    shuffle=True  #in order not to have every similar number in order (00000001111111222222), we shuffle the data
)

model = SimpleNN() #we create an object from our class SimpleNN

criterion = nn.CrossEntropyLoss() #LOSS FUNCTION!!!
#Our model receives an image and returns 10 logits, but we also have the label of each image
#we choose CrossEntropy because we have 10 possible classes, it will evaluate how well our prediction matches the actual label
#by that, we will receive a value of loss, the lower the loss, the less error our prediction has

for images, labels in train_loader: #we obtain 32 images [32, 1, 28, 28] and 32 labels [0,7,4,8,1 ...]

    outputs = model(images)
    #here we call our model, which internally calls the method forward
    #this is the path that it follows:
    #images -> model(images) -> forward(images) -> flatten -> fc1 -> ReLU -> fc2 -> outputs

    loss = criterion(outputs, labels)
    #we compare the outputs from the model with the labels to know how much the model has failed




