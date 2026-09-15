# MNIST-deeplearning
My first deep learning project. It implements a model capable of recognizing handwritten digits

## Features
- Neural network built with PyTorch
- Using MNIST dataset
- 97%+ accuracy

## Technologies
- Python
- PyTorch
- Matplotlib

## Results
Accuracy: 97.5%

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

## BACKPROPAGATION 

Let's take our net as a chain: Image -> fc1 makes calculations -> ReLU modifies some values -> fc2 -> Logits -> Loss 

But PyTorch needs to know, how much each weight has affected the loss. Supposing that our weight changed a bit in fc1,
that affected ReLU, and then fc2, then logits and finally loss. What we are trying to guess here is how much has changed
loss respect to each previous value.

Basically what we do here is going backwards in the "chain" and how much does each operation to the loss,
lets take a simple example where we have Weight, operation A and B and Loss: W -> A -> B -> L

L depends somehow on B, B depends on A, and A depends on W, so we could say L depends on W. Lets calculate with numeric
examples: A = W × 2, B = A + 3, Loss = B², and, for instance, W = 2. The forward path would be the next one:

W = 2 ---x2---> A = 4 ---+3---> B = 7 ---^2---> L = 49

Now, backwards: How much does L changes with respect to B? dL/dB = 2B = 2x7 = 14, and we keep going backwards:
dB/dA = 1, dA/dW = 2, and now we apply chain rule: dL/dW = dL/dB x dB/dA x dA/dW = 14 x 1 x 2 = 28. 28 is our
GRADIENT, as in the previous part, we just need to multiply it by a learning rate and we have our new W.





