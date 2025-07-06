import numpy as np
import matplotlib.pyplot as plt

samples_class_1 = []
while len(samples_class_1) < 10:
    x = np.random.rand()
    y = np.random.rand()
    if y >= x**2:
        samples_class_1.append((x, y, 1))

samples_class_0 = []
while len(samples_class_0) < 10:
    x = np.random.rand()
    y = np.random.rand()
    if y < x**2:
        samples_class_0.append((x, y, 0))

samples = samples_class_1 + samples_class_0
samples_array = np.array(samples)

class_0 = samples_array[samples_array[:, 2] == 0]
class_1 = samples_array[samples_array[:, 2] == 1]

plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', marker='o', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', marker='x', label='Class 1')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Samples Plot')
plt.show()

train_points = samples_array[:, :2]
train_labels = samples_array[:, 2]

# loss function
def loss(y_true, y_pred):
    return 0.5 * np.linalg.norm(y_true - y_pred) ** 2

# activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)
# define neural network
num_input = 2 
num_hidden = 3 
num_output = 1
activation = sigmoid 
learning_rate = 0.1

def model_compile(num_input, num_hidden, num_output, activation, learning_rate):
    W2 = np.random.rand(num_hidden, num_input)
    W3 = np.random.rand(num_output, num_hidden)
    b2 = np.random.rand(num_hidden)
    b3 = np.random.rand(num_output)
    return W2, b2, W3, b3

W2, b2, W3, b3 = model_compile(num_input, num_hidden, num_output, activation, learning_rate)


def forward_pass(X, W2, b2, W3, b3, activation):
    Z2 = np.dot(W2, X) + b2
    A2 = activation(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = activation(Z3)
    return Z2, A2, Z3, A3

Z2, A2, Z3, A3 = forward_pass(train_points[0], W2, b2, W3, b3, activation)


def sigmoid_derivative(x):
    deriv = sigmoid(x) * (1 - sigmoid(x))
    return deriv

def ReLU_derivative(x):
    deriv = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] >= 0:
            deriv[i] = 1
    return deriv

x = np.array([1,2,3,-1,-2,0])


if activation == sigmoid:
    activation_derivative = sigmoid_derivative
elif activation == ReLU:
    activation_derivative = ReLU_derivative

def loss_grad_final_layer(y, A3):
    grad = A3 - y
    return grad 


def backpropagation(X, Y, W2, b2, W3, b3, activation, activation_derivative):
    Z2, A2, Z3, A3 = forward_pass(X, W2, b2, W3, b3, activation)
    d3 = loss_grad_final_layer(Y, A3) * activation_derivative(Z3)
    dW3 = np.outer(d3, A2)
    db3 = d3
    d2 = np.dot(W3.T, d3) * activation_derivative(Z2)
    dW2 = np.outer(d2, X)
    db2 = d2
    
    return dW2, db2, dW3, db3

dW2, db2, dW3, db3 = backpropagation(train_points[0], train_labels[0], W2, b2, W3, b3, activation, activation_derivative)


def model_gradients(X_train, Y_train, W2, b2, W3, b3, activation, activation_derivative):
    num_data_points = len(X_train)
    dW2 = np.zeros_like(W2)
    db2 = np.zeros_like(b2)
    dW3 = np.zeros_like(W3)
    db3 = np.zeros_like(b3)
    for point, label in zip(X_train, Y_train):
        temp_dW2, temp_db2, temp_dW3, temp_db3 = backpropagation(point, label, W2, b2, W3, b3, activation, activation_derivative)
        dW2 += temp_dW2
        db2 += temp_db2
        dW3 += temp_dW3
        db3 += temp_db3
    dW2 = dW2 / num_data_points
    db2 = db2 / num_data_points
    dW3 = dW3 / num_data_points
    db3 = db3 / num_data_points

    return dW2, db2, dW3, db3


dW2, db2, dW3, db3 = model_gradients(train_points, train_labels, W2, b2, W3, b3, activation, activation_derivative)

def model_update(W2, b2, W3, b3, dW2, db2, dW3, db3, learning_rate):
    W2 = W2 - (dW2 * learning_rate)
    b2 = b2 - (db2 * learning_rate)
    W3 = W3 - (dW3 * learning_rate)
    b3 = b3 - (db3 * learning_rate)

    return W2, b2, W3, b3


def model_train(X, Y, W2, b2, W3, b3, activation, activation_derivative, learning_rate, batch_size=32, epochs = 10, metrics=['loss', 'accuracy']):
    num_training_points = len(X)
    for i in range(epochs):
        dW2, db2, dW3, db3 = model_gradients(X, Y, W2, b2, W3, b3, activation, activation_derivative)
        W2, b2, W3, b3 = model_update(W2, b2, W3, b3, dW2, db2, dW3, db3, learning_rate)
        if (i + 1) % 500 == 0:
            total_loss = 0
            num_predictions_correct = 0
            for point, label in zip(X, Y):
                A3 = forward_pass(point, W2, b2, W3, b3, activation)[-1]
                prediction_loss = loss(label, A3)
                label_prediction = np.round(A3)
                if label_prediction == label:
                    num_predictions_correct += 1
                total_loss += prediction_loss
            mean_loss = total_loss / num_training_points
            accuracy = num_predictions_correct / num_training_points
            print(f'Epoch {i + 1}, Loss: {mean_loss}, Accuracy: {accuracy:%}')

    return W2, b2, W3, b3


model_train(train_points, train_labels, W2, b2, W3, b3, activation, activation_derivative, learning_rate, 500)
W2, b2, W3, b3 = model_compile(num_input, num_hidden, num_output, activation, learning_rate)

W2, b2, W3, b3 = model_train(train_points, train_labels, W2, b2, W3, b3, activation, activation_derivative, learning_rate, 5000)
