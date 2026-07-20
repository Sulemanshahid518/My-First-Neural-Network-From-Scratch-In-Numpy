# My-First-Neural-Network-From-Scratch-In-Numpy

## Project Overview:

My neural network is trained on the MNIST dataset to recognize the digits (0-9)

### Tools & Libraries Used:

- **NumPy:** Used heavily for manipulating arrays, calculating dot products, and handling other core math operations.

- **Matplotlib:** Used to plot training loss and the confusion matrix.

- **idx2numpy:** Used to convert the MNIST dataset's idx format into NumPy arrays.

## Project Objective:

The goal of the project is to understand gradients, backpropagation, and forward propagation, as well as how neural networks learn complex patterns in  data.

## Basic Knowledge about **Neural Network:**

<img width="548" height="304" alt="image" src="https://github.com/user-attachments/assets/196409d9-5cd2-43fe-abc6-4b40f86f2487" />




Between any **two layers** of the **Neural Network**, the same process happens

**Stage 1:** Take the input values.

**Stage 2:** Multiply the weights and then sum them. It's actually a dot product. It's a linear value.

**Stage 3:** The Activation Function converts the linear value (dot product) into a non-linear value so that the neural network can learn the complex patterns in data.

## Neural Architecture

Forward Pass:

<img width="634" height="354" alt="image" src="https://github.com/user-attachments/assets/637c1bcc-dd20-46a7-b57b-6a28b9ddca21" />


Firstly, starting with random thetas and taking the  dot product with input features and passing the result to the hidden layer. The hidden layer applies the activation functions (ReLU or sigmoid, or a leaky ReLU) that transform the linear dot product  into non-linear values, and so on for the remaining layers.


**Q:** How to choose the shape of thetas?

thetas_array=(neurons in current layer+bias, neurons in the next layer+bias)
like in our dataset  we have 784 features,  and I chose the number of neurons in the second layer to be 64, so I selected the theta shape (784+1,64+1)
                                      
                                        Input [N x 785] × Theta [785 x 64] = Hidden [N x 64]
Backpropagate:

<img width="2564" height="903" alt="image" src="https://github.com/user-attachments/assets/9c45b2d9-179d-467f-b13f-35eb70d613eb" />



During backpropagation, we compute the error at the output layer and then propagate it backward through the network, layer by layer, by multiplying the error of the current layer by the activations of the previous layer

## Results

**Loss Graph**

<img width="322" height="262" alt="image" src="https://github.com/user-attachments/assets/e6db33c1-79e1-4f08-b7b9-c076e84e47b0" />

**Confusion Matrix**

<img width="335" height="315" alt="image" src="https://github.com/user-attachments/assets/52a871f1-c146-49b7-b7b9-13010115cd8d" />

**Accuracy on Training Data:**  99.4%

**Accuracy on Testing Data:**  97.54%

**Macro F1 Score:** 97.49%

## How to Run the Project:

**1. Install Dependencies:**

Make sure you have Python installed. Then run the following command in your terminal (or command prompt) to install the required libraries:

```bash
pip install numpy matplotlib idx2numpy kagglehub
