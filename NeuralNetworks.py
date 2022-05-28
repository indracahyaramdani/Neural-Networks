import numpy as np
import nnfs
import os
import cv2
import pickle
import copy

nnfs.init()
#Dense layer
class Laye_Dense:

    #Layer initialization
    def __init__(self,n_inputs, n_neurons, weight_regularizer_L1=0,weight_regularizer_L2=0,bias_regularizer_L1=0,bias_regularizer_L2=0):

    







































# Batas Erykka
        # vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \ layer.dweights / \ (np.sqrt(layer.weights_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \ layer.dbiases / \ (np.sqrt(layer.bias_cache) + self.epsilon)

    #call once after any parameter updates
    def post_update_params(self):
    self.iterations += 1

#RMSprop optimizer
class optimizer_RMSprop:

    #initialize optimizer  - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self_decay = decay
        self.iteration = 0
        self.epsilon = epsilon
        self.rho = rho

    #call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \ (1. / (1. + self.decay * self.iterations))

    # update parameters
    def update_params(self, layer):

        #if layer does not contain cache arrays,
        #create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        #update cache with squart current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \ (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \ (1 - self.rho) * layer.dbiases**2


        #vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \ layer.dweights / \ (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biase += -self.current_learning_rate * \ layer.dbiases / \ (np.sqrt(layer.bias_cache) + self.epsilon)  

    # call once any parameter updates
    def post_update_params(self):
        self.iterations += 1

#adam optimizer
class Optimizer_Adam:

    #initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    #call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current.learning_rate = self.learning_rate * \ (1. / (1. + self.decay * self.iterations))
     
    #update parameters
    def update_params(self,layer):

        #if layer does not contain cache arrays,
        #create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        #update momentum with current gradients
        layer.weight_momentums = self.beta_1 * \ layer.weight_momentums + \ (1-self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \ layer.bias_momentums + \ (1-self.beta_1) * layer.dbiases

        #get corrected momentum
        #self.iteration is 0 at first pass
        #and we need to start with 1 here
        weight_momentums_correted = layer.weight_momentums / \ (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_correted = layer.bias_momentums / \ (1 - self.beta_1 ** (self.iterations + 1))

        #update cache with squared current gradients
        layer.weight_cache =  self.beta_2 * layer.weight_cache + \ (1 - self.beta_2) * layer.dweights**2
        layer.weight_cache =  self.beta_2 * layer.bias_cache + \ (1 - self.beta_2) * layer.dbiases**2

        #get corrected cache
        weight_cache_corrected = layer.weight_cache / \ (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \ (1 - self.beta_2 ** (self.iterations + 1))

        #vanilla SGD parameter update + normalization
        #with square rooted cache
        layer.weights += -self.current_learning_rate * \ weight_momentums_corrected / \ (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \ bias.momentums_corrected / \ (np.sqrt(bias_cache_corrected) + self epsilon)

    #call once after any parameter updates
    def post_update_params(self
    self.iterations += 1)

#common loss class
class loss:

    #regularization loss calculation
    def regularization_loss(self):

        #0 by default
        regularization_loss = 0

        #calculate regularization loss
        #iterate all trainable layers
        for layer in self.trainable_layers:

            #L1 regularization - weights
            #calculate only when factor greater than 0
            if layer.weight_regularizer_11 > 0:
                regularization_loss += layer.weight_regularizer_11 * \ np.sum(np.abs(layer.weights))

            #L2 regularization - weights
            if layer.weight_regularizer_12 > 0:
                regularization_loss += layer.weight_regularizer_12 * \ np.sum(layer.weights * \ layer.weights)

            #L1 regularization - biases
            #calculate only when factor greater than 0
            if layer.bias_regularizer_11 > 0:
                regularization_loss += layer.bias_regularizer_11 * \ np.sum(np.abs(layer.biases))

            #L2 regularization - biases
            if layer.bias_regularizer_12 > 0:
                regularization_loss += layer.bias_regularizer_12 * \ np.sum(layer.biases * \ layer.biases)

        return regularization_loss

    #set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    #calculates the data and regularization losses
    #give model output and ground truth values
    def calculate(self, output, y, *, include_regularization=false):

        #calculate sample losses
        sample_losses = self.forward(output, y)

        #calculate mean loss
        data_loss = np.mean(sample_losses)

        #add accumulated sum of losses and simple count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        #if just data loss - return it
        if not inclued_regularization:
            return data_loss

        #return the data and regularization losses
        return data_loss, self.regularization_loss()

    #calculates accumulated loss
    def calculate_accumulated(self, *, include_regularization=false):

        #calculate mean loss
        data loss = self.accumulated_sum / self.accumulated_count

        #if just data loss - return it
        if not include_regularization:
            return data_loss

        #return the data and regularization losses
        return data_loss, self.regularization_loss()

    #reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

































































































































# Batas Widya
























































































# Batas Indra















































































































# Batas Ferna



































# Batas Widya










































#Batas Indra

































