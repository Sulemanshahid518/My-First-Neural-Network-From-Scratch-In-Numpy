# My-First-Neural-Network-From-Scratch-In-Numpy

## Project Overview:

My neural network is trained on the MNIST dataset to recognize the digits (0-9)

### Tools & Libraries Used:

- **NumPy:** Used heavily for manipulating arrays, calculating dot products, and handling other core math operations.

- **Matplotlib:** Used to plot training loss and the confusion matrix.

- **idx2numpy:** Used to convert the MNIST dataset's idx format into NumPy arrays.

- **kagglehub:** Used to download the MNIST dataset directly from Kaggle.

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

Firstly, starting with random thetas and taking the  dot product with input features and passing the result to the hidden layer. The hidden layer applies the activation functions (ReLU or sigmoid,[...])

**Q:** How to choose the shape of thetas?

thetas_array=(neurons in current layer+bias, neurons in the next layer+bias)
like in our dataset  we have 784 features,  and I chose the number of neurons in the second layer to be 64, so I selected the theta shape (784+1,64+1)

Input [N x 785] × Theta [785 x 64] = Hidden [N x 64]

Backpropagate:

<img width="2564" height="903" alt="image" src="https://github.com/user-attachments/assets/9c45b2d9-179d-467f-b13f-35eb70d613eb" />

For every layer, we have to compute two things:

1) **Neuron Error**: This tells us how much blame each neuron has for the incorrect prediction.
2) **Gradient Layer**: This calculates exactly how much we need to adjust the weights to reduce that error.

During backpropagation, we compute the error at the output layer and then propagate it backward through the network, layer by layer, by multiplying the error of the current layer by the activation[...]

## Hyperparameters

Below is a clear table of the hyperparameters used (or commonly used) for this project. Update the values to match the exact settings in your code if they differ.

| Hyperparameter    | Value               | Description |
|-------------------|---------------------|-------------|
| learning_rate     | 0.04                | Step size for gradient descent |
| epochs            | 70                  | Number of full passes through the training set |
| batch_size        | 500                 | Number of samples per gradient update |
| hidden_neurons    | 64                  | Neurons in the hidden layer |
| hidden_neurons_2  | 64                  | Neurons in the second hidden layer |
| activation        | Leaky ReLU          | Activation function for hidden layers |
| output_activation | Softmax             | Activation for the output layer (multiclass) |
| loss_function     | Cross-entropy       | Loss used for training |
| weight_init       | Random normal       | Weight initialization method |


## Results

**Loss Graph**

$$Loss = -\frac{1}{N} \sum_{i=1}^{N} \log(\hat{y}_i)$$

<img width="322" height="262" alt="image" src="https://github.com/user-attachments/assets/e6db33c1-79e1-4f08-b7b9-c076e84e47b0" />

**Confusion Matrix**

<img width="335" height="315" alt="image" src="https://github.com/user-attachments/assets/52a871f1-c146-49b7-b7b9-13010115cd8d" />

**Accuracy on Training Data:**  99.4%

**Accuracy on Testing Data:**  97.54%

**Macro F1 Score:** 97.49%

## Dependencies

The project requires the following Python libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.7+ | Programming language |
| **NumPy** | 1.19+ | Array operations and mathematical computations |
| **Matplotlib** | 3.3+ | Data visualization (plotting loss graphs and confusion matrices) |
| **idx2numpy** | 1.2+ | Convert MNIST idx format to NumPy arrays |
| **kagglehub** | Latest | Download MNIST dataset from Kaggle |


Open your terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run the following command:

```bash
pip install numpy matplotlib idx2numpy kagglehub
```

### **Step 3: Set Up Kaggle API (Optional but Recommended)**

If you want to download the MNIST dataset using `kagglehub`, you may need to set up your Kaggle API credentials:

1. Go to [Kaggle.com](https://www.kaggle.com/) and sign in
2. Navigate to **Settings** → **API** → **Create New API Token**
3. This downloads a `kaggle.json` file
4. Place it in the correct directory:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **macOS/Linux**: `~/.kaggle/kaggle.json`
5. Set permissions (Linux/macOS only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```


Once dependencies are installed, run the notebook using one of the following methods:


#### **Option A: Using Google Colab (No Setup Required)**

1. Upload the notebook to [Google Colab](https://colab.research.google.com/)
2. Run each cell sequentially

#### **Option B: Using VS Code**

1. Install the Jupyter extension in VS Code
2. Open the `.ipynb` file
3. Run each cell by clicking the play button


### **Troubleshooting**

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'idx2numpy'` | Run `pip install idx2numpy` |
| `ModuleNotFoundError: No module named 'kagglehub'` | Run `pip install kagglehub` |
| Kaggle API error when downloading dataset | Set up Kaggle API credentials (see Step 3) or manually download from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) |
| Out of memory error | Reduce `batch_size` in the training loop or `epochs` value |
| Slow training | Use GPU acceleration by running in Google Colab with GPU enabled |

### **Modifying Hyperparameters**

To experiment with different hyperparameters, modify these values in the training cell:

```python
learning_rate = 0.14        # Decrease to train slower, increase for faster learning
epochs = 70                  # Increase for more training iterations
batch_size = 500             # Decrease for more stable gradients, increase for faster training
```
