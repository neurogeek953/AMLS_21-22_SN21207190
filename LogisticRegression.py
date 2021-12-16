#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:07:52 2021

@author: TEB
"""

# NumPy
import numpy as np

# Scikit-learn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# matplotlib
from matplotlib import pyplot as plt

class LogisticRegressor:
    
    def __init__(self,
                 image_dimensions,
                 nb_training_iterations = 2000,
                 cost_sampling_rate = 100,
                 learning_rate = 0.5,
                 verbose = False):
        
        self.weights = np.zeros((image_dimensions))
        self.bias = 0.
        self.learningRate = learning_rate
        self.deltaWeights = 0.
        self.deltaBias = 0.
        self.trainingSetSize = 0
        self.costSamplingRate = cost_sampling_rate
        self.costs = []
        self.iterationCounts = []
        self.trainingIterations = nb_training_iterations
    
    
    def sigmoid(self, x):
        sigmoid_x = 1 / (1 + np.exp(-x))
        return sigmoid_x
    
    def propagate(self, inputs, targets):
        
        # Extract the number of data points
        inputs = inputs.T
        self.training_set_size = inputs.shape[1]
        
        # Calculate the predicted output
        predictions = self.sigmoid(np.dot(self.weights.T, inputs) + self.bias)
        
        # Calculate the Loss
        cost = -1 / self.training_set_size * np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        
        # Compute the gradients
        self.delta_weights = 1 / self.training_set_size * np.dot(inputs, (predictions - targets).T)
        self.delta_bias = 1 / self.training_set_size * np.sum(predictions - targets)
        return cost
    
    def optimize(self, inputs, targets):
        
        # propagate function will run for a number of iterations
        for i in range(self.trainingIterations):
            cost = self.propagate(inputs, targets)
            #Updating the weights and the bias with the gradients
            self.weights = self.weights - self.learningRate * self.delta_weights
            self.bias = self.bias - self.learningRate * self.delta_bias
            
            # Record the cost function value for the cost sampling rate desired
            if i % self.costSamplingRate == 0:
                cost = 0 if np.isnan(cost) == 1 else cost
                self.iterationCounts.append(i)
                self.costs.append(cost)
        
        # convert to numpy arrays
        self.iterationCounts = np.array(self.iterationCounts)
        self.costs = np.array(self.costs)
        plot_counts = self.iterationCounts[0:10]
        plot_costs = self.costs[0:10]
        
        # plot the costs
        plt.figure()
        plt.title("Training Loss")
        plt.xlabel("Iteration Count")
        plt.ylabel("Cross Entropy Loss")
        plt.plot(plot_counts, plot_costs)
    
    def trainLogisticRegressor(self, train_inputs, train_targets):
        
        # train the model
        self.optimize(train_inputs, train_targets)
        # training set predictions
        train_predictions = self.predict(train_inputs)
        print("Training Set Accuracy Logistic Regression: {} %".format(100 - np.mean(np.abs(train_predictions - train_targets)) * 100))
    
    def evaluateLogisticRegressor(self, test_inputs, test_targets, label_encoder = None):
        
        # test set predictions
        test_predictions = self.predict(test_inputs).reshape((-1, 1))
        # make the confusion matrix
        cm = confusion_matrix(test_targets, test_predictions)
        
        if label_encoder is not None:
            test_predictions = label_encoder.inverse_transform(test_predictions)
            test_targets = label_encoder.inverse_transform(test_targets)
            print(classification_report(test_targets, test_predictions))
            labels = list(np.unique(test_targets))
            display_cm = ConfusionMatrixDisplay(cm, display_labels = labels)
            display_cm.plot()
        else:
            # Print the classification report
            print(classification_report(test_targets, test_predictions))
            # plot confusion matrix
            display_cm = ConfusionMatrixDisplay(cm)
            display_cm.plot()
        
        print("The Confusion Matrix Logistic Regressor")
        print(cm)
        # binary classification problem
        accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 0] + cm[0, 1] + cm[1, 1])
        recall = (cm[1, 1]) / ( cm[1, 0] + cm[1, 1])
        specificity = (cm[0, 0]) / ( cm[0, 1] + cm[0, 0])
        
        print("Test Set accuracy Logistic Regressor: {} %".format(accuracy * 100))
        return accuracy, recall, specificity
        
    def predict(self, inputs):
        
        # transpose
        inputs = inputs.T
        # predictions
        predictions = np.zeros((inputs.shape))
        # Flatten the weights
        weights = self.weights.reshape(inputs.shape[0], 1)
        # Compute the models predictive outputs
        predictions_og = self.sigmoid(np.dot(weights.T, inputs) + self.bias)
        # Set the predicted output greater or equal 0.5 and zero otherwise
        predictions = np.around(predictions_og, decimals = 1).astype(int)
        return predictions

if __name__=='__main__':
    
    print("Hello World!")
    lr = LogisticRegressor(150)


