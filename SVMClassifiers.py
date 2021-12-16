#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 20:35:18 2021

@author: TEB
"""

# import NumPy
import numpy as np

# Scikit-Learn imports
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class SupportVectorClassifier:
    
    def __init__(self, grid_parameters):
        
        # initialise the support vector classifier
        self.sVC = SVC(probability = True)
        # Combine Support Vector Machine with grid search
        self.supportVectorClassifier = GridSearchCV(self.sVC, grid_parameters)
    
    def trainSupportVectorClassifier(self, train_inputs, train_targets):
        
        # Train the model
        self.supportVectorClassifier.fit(train_inputs, train_targets)
        print("The best parameters for the model are: ****************************")
        print(self.supportVectorClassifier.best_params_)
        train_predictions = self.predictWithSVC(train_inputs)
        training_accuracy = accuracy_score(train_predictions, train_targets)
        print("Training Set Accuracy Support Vector Classifier: ",
              training_accuracy * 100,
              "%")
        
        
    def predictWithSVC(self, inputs):
        
        # The predict method
        predictions = self.supportVectorClassifier.predict(inputs)
        # Set the predicted output greater or equal 0.5 and zero otherwise
        predictions = np.around(predictions, decimals = 1).astype(int)
        return predictions
    
    def evaluateSupportVectorClassifier(self, inputs, targets, label_encoder = None):
        
        # Evaluate the model
        predictions = self.predictWithSVC(inputs)
        
        # make the confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(predictions)
            targets = label_encoder.inverse_transform(targets)
            print(classification_report(targets, predictions))
            labels = list(np.unique(targets))
            display_cm = ConfusionMatrixDisplay(cm, display_labels = labels)
            display_cm.plot()
        else:
            # Print the classification report
            print(classification_report(targets, predictions))
            # plot confusion matrix
            display_cm = ConfusionMatrixDisplay(cm)
            display_cm.plot()
        
        # print confusion matrix
        print("The Confusion Support Vector Classifier")
        print(cm)
        
        # Metrics
        accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 0] + cm[0, 1] + cm[1, 1])
        recall = (cm[1, 1]) / ( cm[1, 0] + cm[1, 1])
        specificity = (cm[0, 0]) / ( cm[0, 1] + cm[0, 0])
        print("Test Set accuracy Support Vector Classifier: {} %".format(accuracy * 100))
        return accuracy, recall, specificity
        
if __name__ == '__main__':
    
    print("Hello World")
    grid_parameters = {'C':[0.1, 1, 10, 100], 'gamma':[0.0001, 0.001, 0.1, 1], 'kernel':['rbf', 'poly']}
    svc = SupportVectorClassifier(grid_parameters)
    