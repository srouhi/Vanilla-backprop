# Vanilla Neural Network from Scratch

This project implements a basic feedforward neural network with one hidden layer **from scratch using NumPy**, without using any machine learning frameworks like TensorFlow or PyTorch.

The model is trained via **gradient descent and backpropagation**, and uses either the **Sigmoid** or **ReLU** activation function. The network learns to classify synthetic 2D data points as either **above** or **below** the parabola \( y = x^2 \).

---

## Features

- No external ML libraries (pure NumPy)
- Custom implementation of:
  - Forward propagation
  - Backward propagation
  - Loss function
  - Gradient computation
  - Parameter updates via gradient descent
- Option to switch between `sigmoid` and `ReLU` activations
- Visualizes data points and classification decision
- Periodic training checkpoints with **loss and accuracy**


---

## Problem Setup

We generate synthetic 2D data points labeled as:

- Class 1 (label `1`): \( y \geq x^2 \)
- Class 0 (label `0`): \( y < x^2 \)

Each sample is a point \((x, y)\), and the goal is for the network to learn the boundary between the two classes.

---

## Architecture

- **Input Layer**: 2 neurons (x and y)
- **Hidden Layer**: Customizable number of neurons (default: 3)
- **Output Layer**: 1 neuron with output in (0, 1)

---

## Activation Functions

Choose from:
- `sigmoid`: \( \sigma(x) = \frac{1}{1 + e^{-x}} \)
- `ReLU`: \( \text{ReLU}(x) = \max(0, x) \)

Derivatives are implemented manually for backpropagation.

---

## Loss Function

The network is trained using **mean squared error**:

\[
L(y, \hat{y}) = \frac{1}{2} \| y - \hat{y} \|^2
\]

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/vanilla-backprop.git
cd vanilla-backprop
```

### 2. Install requirements
Only numpy and matplotlib are needed:

```bash
pip install numpy matplotlib
```

### 3. Run the project
```bash
python backpropagation_nn.py
```
This will:
- Generate synthetic data
- Train the NN
- print training loss and accuracy
- Display the data plot

---

## Concepts demonstrated:
- How gradient descent works under the hood
- Manual implementation of forward & backward passes
- Role of activation functions and their derivatives
- Basic classification on synthetic data

---

## Author
Shay -Shaghayegh- Rouhi  
Data Science | AI | ML
