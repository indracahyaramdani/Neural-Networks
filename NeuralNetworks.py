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

        #Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        #set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2







































# Batas Erykka



























































































































# Batas Widya
























































































# Batas Indra















































































































# Batas Ferna



































# Batas Widya










































#Batas Indra

































