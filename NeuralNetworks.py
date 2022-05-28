from re import X
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










































#Batas Indra

































