#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 14:56:37 2021

@author: TEB
"""

# NumPy
import numpy as np

# tensorflow imports
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model 

# Create a CNN classifier class
class CNN_Classifier:
    def __init__(self,
                 first_convolutional_layer_arguments,
                 second_convolutional_layer_arguments,
                 third_convolutional_layer_arguments,
                 fourth_convolutional_layer_arguments,
                 fifth_convolutional_layer_arguments,
                 first_pooling_layer_arguments,
                 second_pooling_layer_arguments,
                 third_pooling_layer_arguments,
                 fourth_pooling_layer_arguments,
                 fifth_pooling_layer_arguments,
                 hidden_layer_argument,
                 output_layer_argument,
                 dropout):
        
        
        # set up input size
        self.inputSize = (first_convolutional_layer_arguments[2][0],
                          first_convolutional_layer_arguments[2][0])
        
        # Setup the classifier
        self.Classifier = Sequential()
        
        # Layer 1: 1st Convolutional layer
        self.Classifier.add(Conv2D(first_convolutional_layer_arguments[0],
                                   first_convolutional_layer_arguments[1],
                                   padding = first_convolutional_layer_arguments[2],
                                   input_shape = first_convolutional_layer_arguments[3],
                                   activation = first_convolutional_layer_arguments[4]))
 
        # Layer 2: 1st Pooling Layer
        self.Classifier.add(MaxPool2D(pool_size = first_pooling_layer_arguments))
        
        # Introduce 1st Dropout
        if dropout is True:
            self.Classifier.add(Dropout(0.25))
        
        # Layer 3: 2nd Convolutional Layer
        self.Classifier.add(Conv2D(second_convolutional_layer_arguments[0],
                                   second_convolutional_layer_arguments[1],
                                   padding = second_convolutional_layer_arguments[2],
                                   activation = second_convolutional_layer_arguments[3]))
        
        # Layer 4: 2nd Pooling Layer
        self.Classifier.add(MaxPool2D(pool_size = second_pooling_layer_arguments[0],
                                      strides = second_pooling_layer_arguments[1]))
        
        # Introduce 2nd Dropout
        if dropout is True:
            self.Classifier.add(Dropout(0.25))
        
        
        # Layer 5: 3rd Convolution Layer
        self.Classifier.add(Conv2D(third_convolutional_layer_arguments[0],
                                   third_convolutional_layer_arguments[1],
                                   padding = third_convolutional_layer_arguments[2],
                                   activation = third_convolutional_layer_arguments[3]))
        
        # Layer 6: 3rd Pooling Layer
        self.Classifier.add(MaxPool2D(pool_size = third_pooling_layer_arguments[0],
                                      strides = third_convolutional_layer_arguments[1]))
        
        # Introduce 3rd Dropout
        if dropout is True:
            self.Classifier.add(Dropout(0.3))
        
        # Layer 7: 4th Convolution Layer
        self.Classifier.add(Conv2D(fourth_convolutional_layer_arguments[0],
                                   fourth_convolutional_layer_arguments[1],
                                   padding = fourth_convolutional_layer_arguments[2],
                                   activation = fourth_convolutional_layer_arguments[3]))
        
        # Layer 8: 4th Pooling Layer
        self.Classifier.add(MaxPool2D(pool_size = fourth_pooling_layer_arguments[0],
                                      strides = fourth_pooling_layer_arguments[1]))
        
        # Introduce 4th Dropout
        if dropout is True:
            self.Classifier.add(Dropout(0.3))
        
        # Layer 9: 5th Convolution Layer
        self.Classifier.add(Conv2D(fifth_convolutional_layer_arguments[0],
                                   fifth_convolutional_layer_arguments[1],
                                   padding = fifth_convolutional_layer_arguments[2],
                                   activation = fifth_convolutional_layer_arguments[3]))
        
        # Layer 10: 5th Pooling Layer
        self.Classifier.add(MaxPool2D(pool_size = fifth_pooling_layer_arguments[0],
                                      strides = fifth_convolutional_layer_arguments[1]))
        
        # Introduce 5th Dropout
        if dropout is True:
            self.Classifier.add(Dropout(0.3))
        
        # Layer 11: Flattening Layer
        self.Classifier.add(Flatten())
        
        # Layer 12: 1st Hidden Layer
        self.Classifier.add(Dense(hidden_layer_argument[0],
                                  activation = hidden_layer_argument[1]))
        
        # Introduce 6th Dropout
        if dropout is True:
            self.Classifier.add(Dropout(0.5))
            
        # Layer 13: 2nd Hidden Layer
        self.Classifier.add(Dense(hidden_layer_argument[0] // 2,
                                  activation = hidden_layer_argument[1]))
        
        # Introduce 7th Dropout
        if dropout is True:
            self.Classifier.add(Dropout(0.5))
        
        # Layer 14: Output Layer
        
        if output_layer_argument == 1:
            # 1 Means tumor 0 no tumor
            self.Classifier.add(Dense(units = 1, activation = 'sigmoid'))
            # condition setup for the predict method
            self.outputType = 1
        
        if output_layer_argument == 2:
            # 1 Means tumor 0 no tumor
            self.Classifier.add(Dense(units = 2, activation = 'softmax'))
            # condition setup for the predict method
            self.outputType = 2
        
        
        # four activation units
        if output_layer_argument == 4:
            # make a 4 output node network for multiclass classification
            self.Classifier.add(Dense(units = 4, activation = 'softmax'))
            # condition set up for the predict method
            self.outputType = 4
            
        
        # Compile the CNN
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.Classifier.compile(optimizer = optimizer ,
                                loss = "categorical_crossentropy",
                                metrics=["accuracy"])
        
        print(self.Classifier.summary())
        
    def identifyTumors(self, inputs):
        # the predict method for the model
        predictions_og = self.Classifier.predict(inputs)
        
        if self.outputType == 1:
            # IMPLEMENT THRESHOLD #
            predictions = np.around(predictions_og, decimals = 1).astype(int)
        
        if self.outputType == 2:
            # IMPLEMENT ARGMAX #
            predictions = np.zeros((predictions_og.shape))
            predictions[np.arange(len(predictions)), predictions_og.argmax(1)] = 1
        
        
        if self.outputType == 4:
            # IMPLEMENT ARGMAX #
            predictions = np.zeros((predictions_og.shape))
            predictions[np.arange(len(predictions)), predictions_og.argmax(1)] = 1
        
        return predictions
    
    def saveCNNWeights(self, path):
        # save the CNN model
        self.Classifier.save(path)
    
    
    def loadCNNWeights(self, path):
        # load the CNN model
        self.Classifier = load_model(path)

if __name__=='__main__':
    
    print("Hello World!")
