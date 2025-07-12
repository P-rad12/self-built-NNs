


import numpy as np

# Data
def input_data():
    return np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

def output_data():
    return np.array([[0],
                     [1],
                     [1],
                     [0]])

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Forward pass
def feedforward(X, weights_hidden, bias_hidden, weights_output, bias_output):
    z_hidden = np.dot(X, weights_hidden.T) + bias_hidden
    a_hidden = tanh(z_hidden)

    z_output = np.dot(a_hidden, weights_output) + bias_output
    a_output = sigmoid(z_output)

    return a_hidden, a_output

# Initialize weights and biases
np.random.seed(42)  # for reproducibility

weights_hidden = np.random.uniform(-1, 1, (2, 2))  # 2 hidden neurons, 2 inputs
bias_hidden = np.random.uniform(-1, 1, (1, 2))     # shape (1, 2)

weights_output = np.random.uniform(-1, 1, (2, 1))  # 1 output neuron, 2 inputs
bias_output = np.random.uniform(-1, 1, (1, 1))     # shape (1, 1)

# Training data
X = input_data()
y = output_data()

# Training loop
n_epochs = 10000
learning_rate = 0.1

for epoch in range(n_epochs):
    # Forward pass
    hidden_output, final_output = feedforward(X, weights_hidden, bias_hidden, weights_output, bias_output)

    # Loss
    loss = np.mean((y - final_output) ** 2)

    # Backprop - output to hidden
    d_loss_output = 2 * (final_output - y)
    d_sigmoid = final_output * (1 - final_output)
    output_err_signal = d_loss_output * d_sigmoid

    d_weights_output = np.dot(hidden_output.T, output_err_signal)
    d_bias_output = np.sum(output_err_signal, axis=0, keepdims=True)

    # Backprop - hidden to input
    hidden_error = np.dot(output_err_signal, weights_output.T)
    d_tanh = 1 - hidden_output ** 2
    hidden_error_signal = hidden_error * d_tanh

    d_weights_hidden = np.dot(X.T, hidden_error_signal)
    d_bias_hidden = np.sum(hidden_error_signal, axis=0, keepdims=True)

    # Update parameters
    weights_output -= learning_rate * d_weights_output
    bias_output -= learning_rate * d_bias_output

    weights_hidden -= learning_rate * d_weights_hidden.T
    bias_hidden -= learning_rate * d_bias_hidden

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

# Final prediction
hidden_output, final_output = feedforward(X, weights_hidden, bias_hidden, weights_output, bias_output)
print("\nFinal Predictions:")
print(final_output.round(3))
