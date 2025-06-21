# import numpy as np

# # ----- Data Preparation -----
# def input_data():
#     """Input to the neural network"""
#     return np.array([[0, 0],
#                      [0, 1],
#                      [1, 0],
#                      [1, 1]])

# def output_data():
#     """Expected output for XOR"""
#     return np.array([[0],
#                      [1],
#                      [1],
#                      [0]])

# # ----- Activation Functions -----
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def tanh(x):
#     return np.tanh(x)

# # ----- Feedforward Pass -----
# def feedforward(X, weights_hidden, bias_hidden, weights_output, bias_output):
#     # Hidden layer calculation
#     z_hidden = np.dot(X, weights_hidden.T) + bias_hidden
#     a_hidden = tanh(z_hidden)

#     # Output layer calculation
#     z_output = np.dot(a_hidden, weights_output) + bias_output
#     a_output = sigmoid(z_output)

#     return a_hidden, a_output

# # ----- Initialize Weights and Biases -----
# weights_hidden = np.random.uniform(-1, 1, (2, 2))  # 2 neurons, 2 inputs each
# bias_hidden = np.random.uniform(-1, 1, (2,))       # 2 biases for hidden layer neurons

# weights_output = np.random.uniform(-1, 1, (2, 1))  # 1 output neuron, 2 inputs (from hidden)
# bias_output = np.random.uniform(-1, 1, (1,))       # 1 bias for output neuron

# # ----- Prepare Data -----
# X = input_data()
# y = output_data()

# # ----- Forward Pass -----
# hidden_output, final_output = feedforward(X, weights_hidden, bias_hidden, weights_output, bias_output)

# # ----- Loss Calculation -----
# def calculate_loss(y_true, y_pred):
#     """Calculate Mean Squared Error loss"""
#     return np.mean((y_true - y_pred)**2),((y_true - y_pred)**2)

# loss ,matrix = calculate_loss(y, final_output)

# # ----- Backpropagation -----
# d_loss_output = 2 * (final_output - y)  # Derivative of loss w.r.t. output

# d_sigmoid = final_output * (1 - final_output)  # Derivative of sigmoid

# output_err_signal = d_loss_output*d_sigmoid  # Output layer error signal

# d_weights_output = np.dot(hidden_output.T, output_err_signal)  # Gradient for output weights
# d_bias_output = np.sum(output_err_signal, axis=0)             # Gradient for output bias

# learning_rate = 0.1
# weights_output -= learning_rate * d_weights_output  # Update output weights
# bias_output -= learning_rate * d_bias_output        # Update output bias

# # ----- Backpropagation to Hidden Layer -----
# hidden_error = np.dot(output_err_signal, weights_output.T)  # Error propagated to hidden layer
# d_tanh = 1 - hidden_output**2                               # Derivative of tanh
# hidden_error_signal = hidden_error * d_tanh                 # Hidden layer error signal
# d_weights_hidden = np.dot(X.T, hidden_error_signal)         # Gradient for hidden weights
# d_bias_hidden = np.sum(hidden_error_signal, axis=0, keepdims=True)  # Gradient for hidden bias

# weights_hidden -= learning_rate * d_weights_hidden  # Update hidden weights
# bias_hidden -= learning_rate * d_bias_hidden        # Update hidden bias



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
