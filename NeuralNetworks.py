from inspect import Parameter
from pyexpat import model
from re import X
from statistics import mode
from tkinter import Y
from xml.etree.ElementInclude import include
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



























































































































# Batas Widya
























































































# Batas Indra















































































































# Batas Ferna



































# Batas Widya
 for epoch in range(1,epoch+1):

     #print epoch number
     print(f'epoch:{epoch}')

     #reset accumulated value in loss and accuracy objects
     
     self.loss.new_pass()
     self.accuracy.new_pass()

     # Itterae over steps
     for steps in range(train_steps):

         # If batch size is not set
         # train using one step and full dataset
         if batch_size is None:
             batch_X = X
             batch_y =Y
        
        # Otherwise slice a batch
        else:
            batch_X = X[step*batch_size:(step+1)*batch_size]
            batch_y = y[step*batch_size:(step+1)*batch_size]

        # Perform the forward pass
        output = self.forward(batch_X, training=True)

        #Calculate loss
        data_loss, regularization_loss = \ self.loss.calculate(output, batch_y, include_regularization=True)
        loss = data_loss + regularization_loss

        # Get predictions and calculate an accuracy
        predictions = self.output_layer_activation.predictions(output)
        accuracy = self.accuracy.calcualte(predictions,batch_y)

        # Perform backward pass
        self.backward(output,batch_y)

        #Optimize (update parameters)
        self.optimizer.pre_update_params()
        for layer in self.trainable_layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

        # Print a Summary
        if not step % print_every or step == train_steps-1:
            print(f'step:{step} '+
                  f'acc: {accuracy:.3f}, ' +
                  f'loss : {loss:.3f} (' +
                  f'data_loss : {data_loss:.3f}, ' +
                  f'reg_loss : {regularization_loss:.3f}),' +
                  f'lr:{self.optimizer.current_learning_rate}')
    
    # Get and print epoch loss and accuracy
    epoch_data_loss, epoch_regularization_loss = \ self.loss.calculate_accumulated(include_regularization=True)
    epoch_loss = epoch_data_loss + epoch_regularization_loss
    epoch_accuracy = self.accuracy.calculate_accumulated()

    print(f'training, ' +
          f'acc: {epoch_accuracy:.3f}, ' +
          f'loss: {epoch_loss:.3f}(' +
          f'data_loss : {epoch_data_loss:.3f},' +
          f'reg_loss : {epoch_regularization_loss:.3f}), '+
          f'lr : {self.optimizer.current_learning_rate}')
    
    # If there is the validation data
    if validation_data is not None:

        # Evaluate the model
        self.evaluate(*validation_data,batch_size=batch_size)

# Evaluates the model using passed in dataset
def evaluate(self,X_val,y_val,*, batch_size=None):

    # Default value if batch size is not being set 
    valdation_steps =1

    # Calculate number of steps
    if batch_size is not None:
        validation_steps=len(X_val)//batch_size
        # Dividing round down, If there are some rmaining
        # data but not a full batch, this won't include it 
        # Add '1' to include this not full batch 
        if validation_steps * batch_size < len(X_val):
            validation_steps +=1

    # Reset Accumulated values in loss
    # and accuracy objects
    self.loss.new_pass()
    self.accuracy.new_pass()

    # Iterate over steps
    for step in range(validition_steps):

        # If batch size is not set -
        # train using one step and full dataset
        if batch_size is None:
            batch_X = X_val
            batch_y = y_val
        
        # Otherwise slice a batch
        else:
            batch_X = X_val[
                step*batch_size:(step+1)*batch_size
            ]
            batch_y = y_val[
                step*batch_size:(step+1)*batch_size
            ]
        # Perform the forward pass
        output = self.forward(batch_X,training=False)

        # Calculate the loss
        self.loss.calculate(output, batch_y)

        # Get predictions and calculate an accuracy
        predictions = self.output_layer_activation.predictions(output)
        self.accuracy.calculate(predictions,batch_y)

    # Get and print validation loss and accuracy
    validation_loss = self.loss.calculate_accumulated()
    validation_accuracy = self.accuracy.calculate_accumulated()

    # Print a summary
    print(f'validation, '+
          f'acc: {validation_accuracy:.3f} ' +
          f'loss: {validation_loss:.3f}')

def predict(self, X, *, batch_size=None):

    # Defauult value if batch size is not being set
    prediction_steps = 1

    # Calculate number of steps 
    if batch_size is not None:
        prediction_steps = len(X) // batch_size

        # Dividing rounds down. If there are some remaining
        # data but not a full batch, this won't include it
        # Add '1' to include this not full batch
        if prediction_steps * batch_size < len(X):
            prediction_steps += 1
    # Model outputs
    output = []

    # Itterate over steps
    for steps in range(prediction_steps):

        # If batch size is not set
        # train using one step and full dataset
        if batch_size is None:
            batch_X = X

        # Otherwise slice a batch
        else:
            batch_X = x[step*batch_size:(step+1)*batch_size]

        # Perform the forward pass
        batch_output = self.forward(batch_X, training=False)

        # Append batch prediction to the list of predictions
        output.append(batch_output)
    
    # Stack and return results
    return np.vstack(output)

# Performs forward pass
def forward(self,X,training):

    # Call forward method on the input layer
    # this will set the output property that
    # the first layer in "prev" object is expecting 
    self.input_layer.forward(X, training)

    # Call forward method of every object in a chain
    # Pass output of the previes object as a parameter
    for layer in self.layers:
        layer.forward(layer.prev.output, training)

    # "layer" is now the last object from the list
    # return its output
    return layer.output

def backward(self, output, y):

    # If softmax classifier
    if self.softmax_classifier_output is not None:
        # First call backward method
        # on the combined activation/loss
        # this will set dinputs preperty
        self.softmax_classifier_output.backward(output,y)

        # Since we'll not call bacward method of the last layer
        # which is Softmax activation
        # as we used combined activation/loss
        # object, let's set dinputs in this object
        self.layers[-1].dinputs=\
            self.softmax_classifier_output.dinputs

        # Call backward method going trough
        # all the object but last
        # in reverersed order passing dinputs as a parameter
        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.dinputs)
        
        return

    # First Call Backward method on the loss
    # this will set dinputs property thet the las
    # layer will try to acces shortly
    self.loss.backward(output,y)

    # Call backward method going trough all the objects
    # in reveres order passing dinputs as a parameter
    for layer in reversed(self.layers):
        layer.backward(layer.next.dinputs)

# Retrives and returns parameter of trainable layers
def get_parameters(self):

    # Create a list for parameters
    parameters= []

    # Iterable trainable layers and get their parameter
    for layer in self.trainable_layers:
        parameters.append(layer.get_parameters())

    # Return a list
    return parameters

# Updates the model with new parameters
def set_parameters(self, parameters):

    # Iterate over the parameters and layers
    # and update each layers with each set of the parameters
    for parameter_set, layer in zip(parametes, self.trainable_layers):
        layer.set_parameters(*parameter_set)

def save_parameters(self,path):

    # Open a file in the binary write mode 
    # and save parameters into it 
    with open(path, 'wb') as f:
        picle.dump(self.get_parameters(),f)
# Loads the weights and updates a model instance whit them
def load_parameters(self, path):

    # Open file in the binary-read mode
    # Load weights and update trainable layers
    with open(path, 'rb') as f:
        self.set_parameters(pickle.load(f))

# Save the model 
def save(self, path):

    # Make a feep copy of  current model instance
    model = copy.deepcopy(self)

    # Reset accumulated values in loss and accuracy objects
    model.loss.new_pass()
    model.accuracy.new_pass()

    # Remove data from the input layer
    # and gradients form the loss object
    model.input_layer.__dict__.pop('output',None)
    model.loss.__dict.pop('dinputs',None)

    # For each layer remove inputs, output, and dinputs properties
    for layer in model.layers:
        for property in ['inputs','output','dinputs','dweights','dbiases']:
            layer.__dict__.pop(property, None)

    with open(path, 'wb') as f:
        pickle.dump(model, f)
# Loads and returns a model
@staticmethod
def load(path):
    
    # Open file in the binary-read mode, load model
    with open(path, 'rb') as f:
        model = pickle.load(f)

    # Return a model
    return model



















































#Batas Indra

































