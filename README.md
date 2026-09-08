# MNIST-deeplearning
My first deep learning project. It implements a model capable of recognizing handwritten digits

## Features
- Neural network built with PyTorch
- Using MNIST dataset
- 95%+ accuracy

## Technologies
- Python
- PyTorch
- Matplotlib

## Results
Accuracy: 96.2%

# HOW DOES THE MODEL WORK

## THE MODEL ITSELF

Firstly, it will receive an image from the dataset, represented as [1, 28, 28] which will be flattened into 784 values, our inputs.

Then  comes the first linear layer (fully connected 1), which will reduce our 784 values to 128 (explained deeper in code)

Then we apply ReLU, to add non-linearity to the model

Finally, the second linear layer (fc2) reduces our 128 values into 10 final outputs, which will be our LOGITS, representing the scores for each class


## HOW DOES CROSSENTROPY WORK?

Conceptually, it is done by first, converting logits into probabilities, which is done by softmax.
 f.e. if we have the following logits: [0.9, 0.42, -0.04], softmax turns them into: [0.5, 0.31, 0.19].

CrossEntropy takes this probs and applies -log(prob)
following the previous example, if the correct class was class2, our model would have assigned it with a 19% prob, pretty bad.

CrossEntropy maths = -log(0.19) -> loss = 1.66, HIGH loss, on the other hand, if prob2 would have been 97% f.e., -log(0.97) = 0.03, really good.


## HOW DOES IT LEARN?

The model learns by repeating this pattern: 

Image -> Model prediction -> Loss -> Gradients (Explained below) -> Update parameters 

## GRADIENT'S MATHS
Before we get into the maths, let's experiment a bit:

We will take the smallest model possible, Input = 2, Weight = 3. Prediction = 2x3 = 6 and let's suposse our target is 4.
Using a simple loss function: loss = (prediction - target)^2, loss = 4. This is a very high loss, so, what can we do?
we can just modify our weight, so let's try:

If weight = 3.1 -> Prediction = 6.2 -> Loss = (6.2 - 4)^2 = 4.84, it got higher 

If weight = 2.9 -> Prediction = 5.8 -> Loss = (5.8 - 4)^2 = 3.24, it got better, so now we know we have to reduce weight.
But we cannot do this with every 101,770 parameter on the model, that's why we use gradient

In our example, input = 2, prediction = 2xWeight, so Loss = (2w - 4)^2, which means L depends on W, we want to know how does L change
when W changes, dL/dW, so we derivate: L = (2w - 4)^2 -> dL/dW = 4(2w - 4), with weight = 3, dL/dW = 8. 8 is our GRADIENT.

Gradient is positive, which means if we make W greater, L gets higher as well. If it was negative, it would mean that
the higher the W, the lower the L. Basically, we move opposite to the gradient, positive gradient, we reduce weight
, negative gradient, we increase weight

New weight = Old weight - learningRate(0.1 in this case)*Gradient. For our example, New Weight = 3 - 0.1 * 8

Now, the prediction is 2.2*2 = 4.4, loss = (4.4-4)^2 = 0.16, which is way better, and this is what the model does with every parameter

