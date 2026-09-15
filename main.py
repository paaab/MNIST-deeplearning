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

optimizer = torch.optim.SGD(model.parameters(), lr= 0.1)
#torch.optim is the PyTorch module which has optimizers, and we choose SGD, Stochastic Gradient Descent: New Weight = Old Weight - LearningRate x Gradient
#it will use the gradients calculated by loss.backwards
#model.parameters tells which parameters the model has to update
#learning rate is an hyperparameter. it tells how big is the change that the optimizer does


num_epochs = 5 #this digit is an hyperparameter, its the number of times we are going to repite a pass through the entire dataset

for epoch in range(num_epochs):

    total_loss = 0 #each time we start an epoch, we put total_loss to 0, and we'll add the loss of each batch
    correct = 0
    total = 0

    for images, labels in train_loader: #we obtain 32 images [32, 1, 28, 28] and 32 labels [0,7,4,8,1 ...]

        optimizer.zero_grad()
        #by default, PyTorch calculates every parameter's gradient and saves them, the problem is that it does not replace the values from one batch to another,
        #it accumulates them. zero_grad puts them to 0 before every iteration.

        outputs = model(images)
        #here we call our model, which internally calls the method forward
        #this is the path that it follows:
        #images -> model(images) -> forward(images) -> flatten -> fc1 -> ReLU -> fc2 -> outputs

        loss = criterion(outputs, labels)
        #we compare the outputs from the model with the labels to know how much the model has failed

        loss.backward()
        # PyTorch calculates how the loss changes with respect to every parameter
        # basically, it calculates the gradient of every parameter in the model (101,770)
        # it goes "backwards" because it starts calculating with Loss, and it keeps calculating how that error has affected each previous operation

        optimizer.step()
        #now that we have the gradients, we apply the changes to the weights

        total_loss += loss.item() #loss is a tensor of PyTorch, not a "regular" digit. We can obtain it's value with .item()

        prediction = outputs.argmax(dim=1)
        #outputs has a size like this: [32, 10], we want the 10 columns (dim=1), corresponding to the 10 classes, and we want the biggest of them
        #prediction is an array which contains 32 predictions of our model, we'll then compare them to our labels to count the number of correct predictions

        correct +=  (prediction==labels).sum().item()
        #prediction==label creates an array with bools: True = 1, False = 0, so we can use .sum() on it an get the number of corrects
        #we use .item() to convert a PyTorch tensor into a number

        total += labels.size(0) #usually 32, but in the last batch could be less

    average_loss = total_loss / len(train_loader) #we calculate average loss from each epoch, len(train_loader) is the number of batches
    accuracy = correct/total * 100

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss = {average_loss:.4f}, Accuracy: {accuracy:.2f}")


