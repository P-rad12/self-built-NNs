import numpy as np

def input_data():
    """Input to the neural network"""
    return np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

def output_data():
    """Expected output for XOR"""
    return np.array([[0],
                     [1],
                     [1],
                     [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def feedforward(X, weights_hidden, bias_hidden, weights_output, bias_output):
    # Hidden layer calculation
    z_hidden = np.dot(X, weights_hidden.T) + bias_hidden
    a_hidden = tanh(z_hidden)

    # Output layer calculation
    z_output = np.dot(a_hidden, weights_output) + bias_output
    a_output = sigmoid(z_output)

    return a_hidden, a_output


weights_hidden = np.random.uniform(-1, 1, (2, 2))  # 2 neurons, 2 inputs each
bias_hidden = np.random.uniform(-1, 1, (2,))       # 2 biases for hidden layer neurons

weights_output = np.random.uniform(-1, 1, (2, 1))  # 1 output neuron, 2 inputs (from hidden)
bias_output = np.random.uniform(-1, 1, (1,))       # 1 bias for output neuron

X = input_data()
y = output_data()

hidden_output, final_output = feedforward(X, weights_hidden, bias_hidden, weights_output, bias_output)



def calculate_loss(y_true, y_pred):
    """Calculate Mean Squared Error loss"""

    return np.mean((y_true - y_pred)**2),((y_true - y_pred)**2)

loss ,matrix = calculate_loss(y, final_output)




#backpropagation

d_loss_output = 2 * (final_output - y)


d_sigmoid = final_output * (1 - final_output)



output_err_signal = d_loss_output*d_sigmoid


d_weights_output = np.dot(hidden_output.T, output_err_signal)
d_bias_output = np.sum(output_err_signal, axis=0, keepdims=True)


learning_rate = 0.1
weights_output -= learning_rate * d_weights_output
bias_output -= learning_rate * d_bias_output

