from audioop import ratecv
from turtle import forward
import numpy as np
import nnfs
import os
import cv2
import pickle
import copy

nnfs.init()



#Dense layer
class Layer_Dense:
 
    #Layer initialization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_L1=0, weight_regularizer_L2=0, bias_regularizer_L1=0, bias_regularizer_L2=0):

        #Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        #set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    ##Forward pass
    def forward(self, inputs, training):
        #Remember input values
        self.inputs = inputs
        #Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        
    #Backward pass
    def backward(self, dvaLues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
         # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 *\ self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 *\ self.biases
        # Gradient on values 
        self.dinputs = np.dot(dvalues, self.weights.T)
    
    #Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases
    
    # Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    #Dropout
    class Layer_Dropout :

        # Init 
        def __init__(self, rate):
            #Store rate, we invert it as for example for dropout
            # of 0.1 we need success rate of 0.9
            self.rate = 1 - rate

        # Forward pass
        def forward(self, inputs, training):
            # save input values
            self.inputs = inputs
            # If not in the training mode - return values
            if not training:
                self.output = inputs.copy()
                return
            # Generate and save scaled mask
            self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
            # Apply mask to output values
            self.output = inputs * self.binary_mask
        
        #Backward pass
        def backward(self, dvaLues):
            # Gradient on values
            self.dinputs = dvalues * self.binary_mask

    # Input "layer"
    class Layer_Input:

        # Forward pass
        def forward(self, inputs, trainig):
            self.output = inputs

    # ReLU activation
    class Activation_ReLU:

        # Forward pass
        def forward(self, inputs, training):
            # Remember input values
            self.inputs = inputs
            # Calculate output values from inputs
            self.output = np.maximum(0,inputs)
        
        # Backward pass 
        def backward(self, dvaLues):
            # Since we need to modify original variable,
            # let's make a copy of values first
            self.dinputs = dvalues.copy()

            # Zero gradient where input values were negative
            self.inputs[self.inputs <= 0] = 0

        # Calculate predictions for outputs
        def predictions(self, outputs):
            return outputs

    # Softmax activation 
    class Activation_Softmax:

        # Forward pass
        def forward(self, inputs, training):
            # Remember input values
            self.inputs = inputs

            # Get unnormalized prbabilities
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
            #Normalize them for each sample
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probabilities

        # Backward pass
        def backward(self, dvaLues):

            # Create uninitialized array
            self.dinputs = np.empty_like(dvalues)

            # Enumerate ouputs and gradients
            for index, (single_output, single_dvalues) in \ enumerate(zip(self.output, dvaLues)):
                # Flatten output array
                single_output = single_output.reshape(-1,1)
                # Calculate Jacobian matrix of the output and 
                jacobian_matrix = np.diagflat(single_output) - \ np.dot(single_output, single_output.T)
                # Calculate sample wise gradient
                # and add it to the array of sample gradients
                self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
    # Calculate Predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

    # Sigmoid activation
    class Activation_Sigmoid:

        # Forward pass
        def forward(self, inputs, training):
            # Save input and calculate/save output
            # of the sigmoid function
            self.inputs = inputs
            self.output = 1/(1+np.exp(-inputs))

        # Backward pass
        def backward(self, dvaLues):
            # Derivative - calculates from output of the sigmoid function
            self.dinputs = dvalues * (1 - self.output)*self.output
        
        # Calculate predictions for outputs
        def predictions(self, outputs):
            return (outputs > 0.5) * 1

    # Linear activation
    class Activation_Linear:

        # Forwrad pass
        def forward(self, inputs, training):
            # Just remember values
            self.inputs = inputs
            self.outputs = inputs

        # Backward pass
        def backward(self, dvaLues):
            # derivative is 1, 1 * dvaLues - the chain rule
            self.dinputs = dvalues.copy()

        # Calculate predictions for outputs
        def predictions(self, outputs):
            return outputs

    # SGD optimizer 
    class Optimizer_SGD:

        # Initialize optimizer - set settings,
        # learning rate of 1. is default for this optimizer
        def __init__(self, Learning_rate =1., decay=0., momentum=0.):
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            self.decay = decay
            self.iterations = 0
            self.momentum = momentum

        # Call once before any parameter updates
        def pre_update_params(self):
            if self.decay:
                self.current_learning_rate = self.learning_rate * \ (1./(1.+ self.decay * self.iterations))

        # update parameters
        def update_params(Self, Layer):

            # If we use momentum
            if self.momentum:

                # If layer does not contain momentum arrays, create them 
                # filled with zeros
                if not hasattr(layer, 'weight_momentums'):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    # If there is no momentum array for weight
                    # The array doesn't exist for biases yet either.
                    layer.bias_momentums = np.zeros_like(layer.biases)

                # Build weight update with momentum -take previous
                # Update multiplied by retain factor and update with 
                # current gradients
                weight_updates = \ self.momentum * layer.weight_momentums - \ self.current_learning_rate * layer.dweights
                layer.weight_momentums = weight_updates

                #Build bias updates
                bias_updates = \ self.momentum * layer.bias_momentums - \ self.current_learning_rate * layer.dbiases
                layer.bias_momentums = bias_updates

                # Vanilla SGD updates (as before momentum update)
                else :
                    weight_updates = - self.current_learning_rate * \ layer.dweights
                    bias_updates = - self.current_learning_rate * \ layer.dbiases

                # Update weights and biases using either
                # vanilla or momentum updates
                layer.weights += weight_updates
                layer.biases += bias_updates

            # Call once after any parameter updates
            def post_update_params(self):
                self.iterations += 1
        
    # Adagrad optimizer
    class Optimizer_Adagrad:

        # Initialize optimizer - set settings
        def __init__(self, Learning_rate=1., decay=0., epsilon=1e-7):
            self.learning_rate = Learning_rate
            self.current_learning_rate = Learning_rate 
            self.decay = decay
            self.iterations = 0
            self.epsilon = epsilon
        
        # Call once before any parameter updates
        def pre_update_params(self):
            if self.decay:
                self.current_learning_rate = self.learning_rate*\(1./(1. + self.decay * self.iterations))
        
        # Update parameters
        def update_params(self, Layer):

            # If layer does not contain cache arrays,
            # Create them filled with zeros
            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2







        













































# Batas Widya










































#Batas Indra

































