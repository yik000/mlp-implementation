# Generate 500 vectors containing 4 components each. (2,000 numbers)
# The value of each component should be a random number [-1, 1].
# These will be the input vectors.
# The corresponding output for each vector should be the sin() of a combination of the components.
# Specifically, for inputs:
#   [x1 x2 x3 x4]
# the (single component) output should be:
# sin(x1-x2+x3-x4)
#
# Train an MLP with 4 inputs, at least 5 hidden units, and 1 output on 400 of the examples.
# Keep remaining 100 examples for testing.


from mlp import MLP
from math import sin
import numpy as np

# Generate examples
examples = []
for i in range(500):
    arr = []
    inputs = np.random.uniform(-1,1,4)
    outputs = []
    outputs.append( sin(inputs[0]-inputs[1]+inputs[2]-inputs[3]) )
    arr.append(inputs)
    arr.append(outputs)
    examples.append(arr)

# Split examples into train/test (400/100)
train = examples[:400]
test = examples[400:]

# Train MLP on examples
# set hyperparameters
max_epochs = 50
learning_rate = 0.001
W_update_interval = 20
n_hidden_units = 6

# Initialise MLP - 4 inputs, n hidden units, 1 output
NN = MLP(4, n_hidden_units, 1)

print("#"*50)
print("##########", "Running Test 2", "#"*24)
print("#")
print(f"Number of Epochs: {max_epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Weight update interval: {W_update_interval} (examples)")
print(f"Number of hidden units: {n_hidden_units}")
print("#")

# Randomise the weights
NN.randomise()

# Training - on 400 examples
train_count = 0
for e in range(max_epochs):
    error = 0
    for p in train: 

        NN.forward(p[0])
        train_count += 1
        
        error += NN.backward(p[1])

        if train_count % W_update_interval == 0:
            NN.update_weights(learning_rate)

    error = 0.5 * error
    print(f"Error at epoch {e+1} is {error}")

print("#")

# Testing - on 100 examples
error = 0
for p in test:

    NN.forward(p[0])
    error += NN.backward(p[1])

error  = 0.5 * error
print(f"Error on test set: {error}")

print("#")
print("#"*50)