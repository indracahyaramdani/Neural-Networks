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
        


        













































# Batas Widya










































#Batas Indra

































