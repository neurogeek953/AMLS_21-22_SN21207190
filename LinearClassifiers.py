#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 15:12:56 2021

@author: TEB
"""


# import the logistic regressor
from LogisticRegression import LogisticRegressor

# import the SVM with grid search
from SVMClassifiers import SupportVectorClassifier

class LinearClassifiers:
    
    def __init__(self,
                 flat_image_dimension_lr,
                 flat_image_dimension_svc,
                 grid_parameters,
                 lr_nb_training_iterations = 2000,
                 lr_cost_sampling_rate = 100,
                 lr_learning_rate = 0.5,
                 lr_verbose = False):
        
        # initialize the models
        self.LR = LogisticRegressor(flat_image_dimension_lr,
                                    lr_nb_training_iterations,
                                    lr_cost_sampling_rate,
                                    lr_learning_rate,
                                    lr_verbose)
        
        self.SVCGS = SupportVectorClassifier(grid_parameters)
        
        ## LR
        # Accuracy
        self.LRAccuracy = 0.
        # Recall
        self.LRRecall = 0.
        # Specificity
        self.LRSpecificity = 0.
        ## SVCGS
        # Accuracy
        self.SVCGSAccuracy = 0.
        # Recall
        self.SVCGSRecall = 0.
        # Specificity
        self.SVCGSSpecificity = 0.
    
    def trainClassifiers(self, inputs, targets):
        
        # train the classifiers
        self.LR.trainLogisticRegressor(inputs, targets)
        self.SVCGS.trainSupportVectorClassifier(inputs, targets)
        
    def identifiyTumors(self, inputs):
        
        # make the predictions using the classifiers
        LR_predictions = self.LR.predict(inputs)
        SVCGS_predictions = self.SVCGS.predictWithSVC(inputs)
        return LR_predictions, SVCGS_predictions
    
    def evaluateLinearClassifiers(self, inputs, targets):
        
        # test the models and get their accuracies and costs
        self.LRAccuracy, self.LRRecall, self.LRSpecificity = self.LR.evaluateLogisticRegressor(inputs,
                                                                                               targets)
        self.SVCGSAccuracy, self.SVCGSRecall, self.SVCGSSpecificity = self.SVCGS.evaluateSupportVectorClassifier(inputs,
                                                                                                                 targets)
        
        