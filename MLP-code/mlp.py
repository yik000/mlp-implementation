import numpy as np

class MLP:

    def __init__(self, inputs, hidden_units, outputs) -> None:
        
        self.NI = inputs #number of inputs
        self.NH = hidden_units #number of hidden units
        self.NO = outputs #number of outputs
        self.W1 = [] #[[]] #array of weights in lower layer - NI arrays of NH elements
        self.W2 = [] #[[]] #array of weights in upper layer - NH arrays of NO elements
        self.dW1 = [[0]*self.NH]*self.NI #[[]] #array of weight changes to be applied to W1
        self.dW2 = [[0]*self.NO]*self.NH #[[]] #array of weight changes to be applied to W2
        self.Z1 = [0] * self.NH #array of activations for lower layer
        self.Z2 = [0] * self.NO #array of activations for upper layer
        self.I = [] # array of input values
        self.H = [0] * self.NH #array of values of hidden neurons are stored (for computing dW2)
        self.O = [0] * self.NO #array of output values


    def randomise(self):
        """
        Initialises W1 and W2 to small random values.
        Sets dW1 and dW2 to all zero.
        """
        # set weights in W1 and W2 in range [-0.1, 0.1]
        for i in range(self.NI):
            self.W1.append(np.random.uniform(-0.1, 0.1, self.NH))
        for i in range(self.NH):
            self.W2.append(np.random.uniform(-0.1, 0.1, self.NO))
        
        # set weight changes to zero
        self.dW1 = [[0]*self.NH]*self.NI
        self.dW2 = [[0]*self.NO]*self.NH


    def forward(self, example_inputs):
        """
        Forward pass.
        Input vector I is processed to produce an output stored in O.
        """
        self.I = example_inputs
        
        # get activation for layer 1
        for h in range(self.NH):
            activation = 0
            for i in range(self.NI):
                activation += self.W1[i][h] * example_inputs[i]
            self.H[h] = activation
            self.Z1[h] = activation

        # get activations for layer 2 using activations from layer 1
        outputs = []
        for o in range(self.NO):
            activation = 0
            for h in range(self.NH):
                activation += self.W2[h][o] * self.Z1[h]
            self.Z2[o] = activation
            outputs.append(activation)
            
        # add output vector
        self.O = outputs


    def backward(self, target):
        """
        Backward pass.
        Target t is compared with output O, deltas are computed for the upper layer, and are multiplied by the inputs to the layer (H) to produce weight updates dW2.
        Then deltas are produced for lowest layer, and are multiplied by the inputs to the layer to produce weight updates dW1.
        Returns error on the example.
        """
        # Comput deltas for upper layer
        deltas1 = []
        for o in range(self.NO): 
            delta = (target[o] - self.O[o])
            deltas1.append(delta)

        # Multiply inputs to layer (H) by deltas to produce weight updates dW2
        for h in range(self.NH):
            for w in range(self.NO):
                self.dW2[h][w] += deltas1[w] * self.Z1[h]

        # Produce deltas for lower layer
        deltas2 = []
        for h in range(self.NH):
            delta = 0
            for o in range(self.NO):
                delta += deltas1[o] * self.W2[h][o]
            deltas2.append(delta)

        # Multiply inputs to layer by deltas to produce weight updates dW1
        for i in range(self.NI):
            for w in range(self.NH):
                self.dW1[i][w] += deltas2[w] * self.I[i]
        
        # Calculate and return the error - Compare Target t with output O
        error = 0
        for o in range(self.NO):
            error += (target[o] - self.O[o]) ** 2
        
        return error


    def update_weights(self, learning_rate):
        """
        Updates the weights in each layer.
        """
        # update weights - Learning Algorithm
        for i in range(self.NI):
            for w in range(self.NH):
                self.W1[i][w] += learning_rate * self.dW1[i][w]
        
        for h in range(self.NH):
            for w in range(self.NO):
                self.W2[h][w] += learning_rate * self.dW2[h][w]

        # reset weight change arrays
        self.dW1 = [[0]*self.NH]*self.NI
        self.dW2 = [[0]*self.NO]*self.NH
        