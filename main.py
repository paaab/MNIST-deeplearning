from torchvision import datasets
from torchvision import transforms

transform = transforms.ToTensor() #neural networks work with tensors instead of images, transforms images into tensors,
                                  #dividing each pixel by 255 in order to get values between 0.0 and 1.0

train_dataset = datasets.MNIST(
    root="data", #saves dataset in "data"
    train=True, #MNIST is divided in 2 sets, train=True takes the train set (60,000 samples)
    download=True, #downloads if needed, only the first time the project is executed
    transform=transform #uses the transform defined previously for each image
)



