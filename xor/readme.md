 Building and Training a Neural Network for XOR – Key Steps and Concepts
 1. Data Representation
 • 
• 
• 
Inputs and outputs for XOR are represented as lists of lists:
 inputs = [[0,0], [0,1], [1,0], [1,1]]
 outputs = [[0], [1], [1], [0]]
 2. Network Structure
 • 
• 
• 
The network has three layers:
 Input Layer: 2 neurons (
 x1 , 
x2 )
 Hidden Layer: 2 neurons (
 h1 , 
h2 )
 • 
• 
• 
Output Layer: 1 neuron (
 o )
 Every input connects to every hidden neuron; every hidden neuron connects to the output neuron.
 Each weight and bias is unique to its connection/neuron.
 3. Forward Pass
 • 
• 
• 
• 
• 
◦ 
◦ 
◦ 
◦ 
◦ 
For each input:
 Compute weighted sums for each hidden neuron:
 z1 = w1_1*x1 + w1_2*x2 + b1
 z2 = w2_1*x1 + w2_2*x2 + b2
 Apply sigmoid activation to each weighted sum:
 h1 = sigmoid(z1)
 h2 = sigmoid(z2)
 Compute output neuron’s weighted sum:
 zo = v1*h1 + v2*h2 + bo
 Apply sigmoid activation to output sum:
 o = sigmoid(zo)
 ◦ 
4. Error Calculation
 • 
• 
Calculate the error for each output:
 error = actual_output - predicted_output
 5. Backpropagation (Calculating Contributions)
 • 
• 
• 
• 
Compute the gradient at the output neuron:
 go = error * sigmoid_derivative(output)
 Propagate the error back to the hidden neurons:
 For hidden neuron 1: 
hidden1_error = go * v1
 1
For hidden neuron 2: 
• 
• 
• 
• 
• 
• 
hidden2_error = go * v2
 Compute sigmoid derivative at each hidden neuron.
 Compute gradients for each hidden neuron:
 gh1 = hidden1_error * sigmoid_derivative(h1)
 gh2 = hidden2_error * sigmoid_derivative(h2)
 These gradients show how much each neuron (and their associated weights and biases) contributed
 to the output error.
 6. Weight and Bias Update
 • 
• 
• 
• 
Update each variable using its calculated gradient (contribution) and the learning rate:
 new_weight = old_weight + learning_rate * gradient * input
 new_bias = old_bias + learning_rate * gradient * 1
 This is done for all weights and biases in both layers.
 7. Repeat Training
 • 
• 
Repeat the process (forward, error, backprop, update) for every sample, for multiple epochs.
 This allows the network to learn and minimize error on the XOR dataset.
 Extra Insights:
 • 
• 
• 
No weights or biases are shared between neurons or layers; each is updated independently.
 The gradient’s sign tells you the direction to update each parameter to reduce error.
 This process splits up and “assigns blame” for the error across all weights and biases, so each can be
 nudged in the best direction