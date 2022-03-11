# Train MLP with 2 inputs, 3-4+ hidden units and 1 output.
# Examples:
#   ((0, 0), 0)
#   ((0, 1), 1)
#   ((1, 0), 1)
#   ((1, 1), 0)

########## NOTES
# NH =3-4+ and you might experience easier trainig

from mlp import MLP

# set examples
examples = [
    [[0,0], [0]],
    [[0,1], [1]],
    [[1,0], [1]],
    [[1,1], [0]]
]

# Train MLP on examples
# set hyperparameters
max_epochs = 50
learning_rate = 0.1
W_update_interval = 4
n_hidden_units = 4

# Initialise MLP - 2 inputs, n hidden units, 1 output
NN = MLP(2, n_hidden_units, 1)

print("#"*50)
print("##########", "Running Test 1", "#"*24)
print("#")
print(f"Number of Epochs: {max_epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Weight update interval: {W_update_interval} (examples)")
print(f"Number of hidden units: {n_hidden_units}")
print("#")

# Randomise the weights
NN.randomise()

# Train MLP
train_count = 0
for e in range(max_epochs):
    error = 0
    for p in examples:
        
        NN.forward(p[0])
        train_count += 1

        error += NN.backward(p[1])
        
        if train_count % W_update_interval == 0:
            NN.update_weights(learning_rate)

    error = 0.5 * error
    print(f"Error at epoch {e+1} is {error}")

print("#")
print("#"*50)