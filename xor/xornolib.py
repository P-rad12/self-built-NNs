import random
import math

# Data
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
outputs = [
    [0],
    [1],
    [1],
    [0]
]

# Activation functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)

# Feedforward function
def feedforward(x1, x2, w1_1, w1_2, w2_1, w2_2, b1, b2, v1, v2, bo):
    z1 = w1_1 * x1 + w1_2 * x2 + b1
    h1 = sigmoid(z1)
    z2 = w2_1 * x1 + w2_2 * x2 + b2
    h2 = sigmoid(z2)
    zo = v1 * h1 + v2 * h2 + bo
    o = sigmoid(zo)
    return h1, h2, o

# Weight and bias initialization
w1_1 = random.uniform(-1, 1)
w1_2 = random.uniform(-1, 1)
w2_1 = random.uniform(-1, 1)
w2_2 = random.uniform(-1, 1)
b1 = random.uniform(-1, 1)
b2 = random.uniform(-1, 1)
v1 = random.uniform(-1, 1)
v2 = random.uniform(-1, 1)
bo = random.uniform(-1, 1)

# Hyperparameters
learning_rate = 0.5
epochs = 10000

# Training loop
for epoch in range(epochs):
    for i in range(len(inputs)):
        x1, x2 = inputs[i]
        target = outputs[i][0]

        # --- Forward pass ---
        h1, h2, o = feedforward(x1, x2, w1_1, w1_2, w2_1, w2_2, b1, b2, v1, v2, bo)

        # --- Error calculation ---
        error = target - o

        # --- Backpropagation (output layer) ---
        go = error * sigmoid_derivative(o)  # output gradient

        # --- Backpropagation (hidden layer) ---
        hidden1_error = go * v1
        hidden2_error = go * v2

        gh1 = hidden1_error * sigmoid_derivative(h1)  # gradient for h1
        gh2 = hidden2_error * sigmoid_derivative(h2)  # gradient for h2

        # --- Update output layer weights and bias ---
        v1 += learning_rate * go * h1
        v2 += learning_rate * go * h2
        bo += learning_rate * go * 1

        # --- Update hidden layer weights and biases ---
        w1_1 += learning_rate * gh1 * x1
        w1_2 += learning_rate * gh1 * x2
        b1   += learning_rate * gh1 * 1

        w2_1 += learning_rate * gh2 * x1
        w2_2 += learning_rate * gh2 * x2
        b2   += learning_rate * gh2 * 1

# --- Test after training ---
for i in range(len(inputs)):
    x1, x2 = inputs[i]
    h1, h2, o = feedforward(x1, x2, w1_1, w1_2, w2_1, w2_2, b1, b2, v1, v2, bo)
    print(f"Input: {x1}, {x2} => Predicted: {round(o)}, Actual: {outputs[i][0]}, Raw Output: {o:.3f}")
