#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:22:06 2021

@author: TEB
"""

# import Numpy
import numpy as np

# Pandas
import pandas as pd

# scikit-learn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# imblearn
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE

# matplotlib
from matplotlib import pyplot as plt

# import python-openCV
import cv2

# import the CNN classifier class
from CNN import CNN_Classifier

# preprocess to data to get the inputs and outputs
def preprocess_data(target_path, sampling_strategy = None):
    
    # Targets
    labels = pd.read_csv(target_path)
    labels = labels.to_numpy()
    binary_labels = []
    binary_labels_str = []
    categories = {}
    binary_sampler = None
    multiclass_sampler = None
    file_encoder = LabelEncoder()
    
    ### Set up unbalanced sampling
    
    if sampling_strategy == "smote":
        binary_sampler = SMOTE(sampling_strategy = 1., random_state = 42)
        multiclass_sampler = SMOTE(sampling_strategy = "minority", random_state = 42)
    
    if sampling_strategy == "adasyn":
        binary_sampler = ADASYN(sampling_strategy = 1., random_state = 42)
        multiclass_sampler = ADASYN(sampling_strategy = "minority", random_state = 42)
    
    # Inputs
    img_file_names = labels[:, 0]
    img_file_codes = file_encoder.fit_transform(img_file_names).reshape(-1, 1)
    
    # Create the binary labels
    for i in range(0, labels.shape[0]):
        
        # examine the items in the different categories
        if labels[i, 1] not in categories:
            categories[labels[i, 1]] = 1
        else:
            categories[labels[i, 1]] += 1
        
        # prepare the string binary labels
        if labels[i, 1] == "no_tumor":
            binary_labels_str.append("no_tumor")
        else:
            binary_labels_str.append("tumor")
    
    # binary labels convert to numpy
    binary_labels_og = np.array(binary_labels_str).reshape(-1, 1)
    le_binary = LabelEncoder()
    binary_labels_og = le_binary.fit_transform(binary_labels_og).reshape(-1, 1)
    
    # resampple the binary datsets
    if binary_sampler is not None:
        img_file_codes_binary, binary_labels = binary_sampler.fit_resample(img_file_codes, binary_labels_og)
        img_file_names_binary = file_encoder.inverse_transform(img_file_codes_binary)
    else:
        img_file_names_binary = img_file_names 
        binary_labels = binary_labels.reshape(-1, 1)
    
    multiclass_labels_og = np.array(labels[:, 1]).reshape(-1, 1)
    
    # resample the multiclass datasets
    if multiclass_sampler is not None:
        img_file_codes_multiclass, multiclass_labels = multiclass_sampler.fit_resample(img_file_codes, multiclass_labels_og)
        img_file_names_multiclass = file_encoder.inverse_transform(img_file_codes_multiclass)
    else:
        img_file_names_multiclass = img_file_names
        multiclass_labels = multiclass_labels
    
    # Label encoding
    le = LabelEncoder()
    multiclass_labels_og = le.fit_transform(multiclass_labels_og).reshape(-1, 1)
    multiclass_labels = le.transform(multiclass_labels).reshape(-1, 1)
    
    # One hot encoding
    ohe4_cat = OneHotEncoder(handle_unknown = "ignore", sparse = False)
    ohe2_cat = OneHotEncoder(handle_unknown = "ignore", sparse = False)
    binary_ohe_labels_og = ohe2_cat.fit_transform(binary_labels_og)
    binary_ohe_labels = ohe2_cat.transform(binary_labels.reshape(-1, 1))
    multiclass_labels_og = ohe4_cat.fit_transform(multiclass_labels_og)
    multiclass_labels = ohe4_cat.transform(multiclass_labels.reshape(-1, 1))
    
    # return the dictionary of categories and one-hot encoded 
    return categories, multiclass_labels, img_file_names_multiclass, labels, img_file_names_binary, binary_labels, binary_ohe_labels, ohe4_cat, ohe2_cat, img_file_names, binary_labels_og, binary_ohe_labels_og, multiclass_labels_og, le, le_binary

# First Convolutional layer parameters
def select_1st_convolutional_layer_arguments(filters,
                                             kernel_size,
                                             padding = "Same",
                                             input_shape = (150, 150, 3),
                                             activation_function = 'relu'):
    # set up arguments for the first convolutional layer
    output_neurons = filters
    window_size = kernel_size
    padding = padding
    activation_function = activation_function
    input_shape = input_shape
    return [output_neurons, window_size, padding, activation_function, input_shape]

# Convolutional layer parameters excluding the first layer
def select_2nd_convolutional_layer_arguments(filters,
                                             kernel_size,
                                             padding = 'Same',
                                             activation_function = 'relu'):
    # set up arguments for the all convolutional layers except the first
    output_neurons = filters
    window_size = kernel_size
    padding = padding
    activation_function = activation_function
    return [output_neurons, window_size, padding, activation_function]

# set up pooling layer arguments
def select_pooling_layer_arguments(poolX, poolY):
    return (poolX, poolY)

# set up pooling layer arguments
def select_pooling_layer_window_size_arguments(poolX, poolY, strideX, strideY):
    window_size = (poolX, poolY)
    stride = (strideX, strideY)
    return[window_size, stride]

# set up hidden layer parameters
def select_hidden_layer_arguments(nb_neurons, activation_function = 'relu'):
    return [nb_neurons, activation_function]

# Image preprocessing
def get_image(file_path, img_file_name, img_size = 150):
    # read image
    img = cv2.imread(file_path + "/" + img_file_name)
    # resize the image (Pyramid Reduction Method)
    img = cv2.resize(img, (img_size, img_size))
    # mormalize image pixels
    img = img / 255.
    return img

# Preprocess all images
def preprocess_input_data(input_file_names, file_path, image_dimension_xy):
    
    # set an array for the inputs
    inputs = []
    
    # gather the data from individual pixels
    for i in range(0, len(input_file_names)):
        img = get_image(file_path, input_file_names[i], image_dimension_xy)
        inputs.append(img)
    
    # convert to numpy array
    inputs = np.array(inputs)
    return inputs

# Plot the progress of the CNN's training
def plot_model_history(history):
    
    # Get the variables
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    # Plot the model accuracy
    plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Test Set'], loc='upper left')
    plt.show()
    
    # Plot the model loss
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Test Set'], loc='upper left')
    plt.show()
    
## The CNN taining function
def train_and_crossvalidate(input_file_names,
                            targets,
                            file_path,
                            nb_folds,
                            batch_size,
                            nb_epochs,
                            verbosity,
                            image_dimension_xy,
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
    
    # Tables to store the accuracies
    acc_per_fold = []
    loss_per_fold = []
    
    # set up the best score to infinity
    best_score = [float("inf"), float("-inf")]
    
    # set up the best_classifier and best_history variables
    best_classifier = None
    best_history = None
    
    
    # set up the inputs and the outputs
    inputs = preprocess_input_data(input_file_names,
                                   file_path,
                                   image_dimension_xy)
    # set up the targets
    targets = targets
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=nb_folds, shuffle=True)
    
    # K-fold Cross Validation model evaluation
    fold_number = 1
    folds = np.linspace(1, nb_folds, nb_folds)
    
    # Iterate over the k-folds
    for train, test in kfold.split(inputs, targets):
        
        # Define the model architecture for the CNN
        classifier = CNN_Classifier(first_convolutional_layer_arguments,
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
                                    dropout)
        
        # Generate a print to indicate the number of the current k-fold
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_number} ...')
        
        # prepare the data for the CNN
        X_train, X_val, Y_train, Y_val = train_test_split(inputs[train], targets[train], test_size = 0.2, random_state=42)
        
        # Fit the data to the model
        history = classifier.Classifier.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs = nb_epochs)
        
        
        # Generate generalization metrics
        scores = classifier.Classifier.evaluate(inputs[test], targets[test], verbose = 0)
        print(f'Score for fold {fold_number}: {classifier.Classifier.metrics_names[0]} of {scores[0]}; {classifier.Classifier.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        
        # Increase fold number
        fold_number = fold_number + 1
        
        # Generate generalization metrics
        score = classifier.Classifier.evaluate(inputs[test], targets[test], verbose = 0)
        
        # Update the best parameters
        if score[1] > best_score[1]:
            best_score[0] = score[0]
            best_score[1] = score[1]
            best_classifier = classifier
            best_history = history
            
        # Print the test losses
        print(f'Current Test loss: {score[0]} / Test accuracy: {score[1]}')
        print(f'Best Test loss: {best_score[0]} / Test accuracy: {best_score[1]}')
    
    
    
    # Plot epochs of the best CNN
    plot_model_history(best_history)
    
    # Test Loss over K-Folds
    plt.figure()
    plt.title("Test Loss per Fold")
    plt.xlabel("Fold number")
    plt.ylabel("Test Loss")
    plt.plot(folds, loss_per_fold)
    
    # Test Accuracy over K-Folds
    plt.figure()
    plt.title("Test Accuracy per Fold")
    plt.xlabel("Fold number")
    plt.ylabel("Test Accuracy in per cent [%]")
    plt.plot(folds, acc_per_fold)
    
    return best_classifier

# Test the CNN's performance
def test_and_evaluate(input_file_names,
                      file_path, labels,
                      model,
                      image_dimension_xy,
                      label_encoder = False,
                      ohe = False):
    
    # set up the inputs and the outputs
    inputs = preprocess_input_data(input_file_names, file_path, image_dimension_xy)
    # get the test labels
    targets = labels
    # get the predictions
    predictions = model.identifyTumors(inputs)
    
    # binary classification 1 node network
    if model.outputType == 1:
        # get the original labels for the targets
        targets = label_encoder.inverse_transform(targets)
        # get the original labels for the predictions
        predictions = label_encoder.inverse_transform(predictions)
        # make the confusion matrix
        cm = confusion_matrix(targets, predictions)
        # print the classification scores
        print(classification_report(targets, predictions))
        print("The Confusion Matrix")
        print(cm)
        # Get the labels
        labels = list(np.unique(targets))
        # plot the confusion matrix
        display_cm = ConfusionMatrixDisplay(cm, display_labels = labels)
        display_cm.plot()
        # compute accuracy
        accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 0] + cm[0, 1] + cm[1, 1])
        # compute recall
        recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        # compute specificity
        specificity = cm[0, 0] / (cm[0, 1] + cm[0, 0])
        # get the performance output
        performance = [accuracy, recall, specificity, -1, -1, -1]
        return performance
    
    # binary classification 2 node network
    if model.outputType == 2:
        # get the true labels for targets
        targets = ohe.inverse_transform(targets)
        # gets the true labels for the predictions
        predictions = ohe.inverse_transform(predictions)
        # get the original labels for the targets
        targets = label_encoder.inverse_transform(targets)
        # get the original labels for the predictions
        predictions = label_encoder.inverse_transform(predictions)
        # make the confusion matrix
        cm = confusion_matrix(targets, predictions)
        # print the classification scores
        print(classification_report(targets, predictions))
        # print the confusion matrix
        print("The Confusion Matrix")
        print(cm)
        # Get the labels
        labels = list(np.unique(targets))
        # plot the confusion matrix
        display_cm = ConfusionMatrixDisplay(cm, display_labels = labels)
        display_cm.plot()
        # compute accuracy
        accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 0] + cm[0, 1] + cm[1, 1])
        # compute recall
        recall = (cm[1, 1]) / ( cm[1, 0] + cm[1, 1])
        # compute specificity
        specificity = (cm[0, 0]) / ( cm[0, 1] + cm[0, 0])
        # get the performance output
        performance = [accuracy, recall, specificity, -1, -1, -1]
        return performance
    
    # multiclass classification problem
    if model.outputType == 4:
        # get the true labels for the targets
        targets = ohe.inverse_transform(targets)
        # get the true labels for the predictions
        predictions = ohe.inverse_transform(predictions)
        # get the original labels for the targets
        targets = label_encoder.inverse_transform(targets)
        # get the original labels for the predictions
        predictions = label_encoder.inverse_transform(predictions)
        # make the confusion matrix
        cm = confusion_matrix(targets, predictions)
        # print the classification report
        print(classification_report(targets, predictions))
        # print the confusion matrix
        print("The Confusion Matrix")
        print(cm)
        # Get the labels
        labels = list(np.unique(targets))
        # plot the confusion matrix
        display_cm = ConfusionMatrixDisplay(cm, display_labels = labels)
        display_cm.plot(xticks_rotation = "vertical")
        # compute accuracy
        accuracy = (cm[0, 0] + cm[1, 1] + cm[2, 2] + cm[3, 3]) / (cm[0, 0] + cm[0, 1] + cm[0, 2] + cm[0, 3] + cm[1, 0] + cm[1, 1] + cm[1, 2] + cm[1, 3] + cm[2, 0] + cm[2, 1] + cm[2, 2] + cm[2, 3] + cm[3, 0] + cm[3, 1] + cm[3, 2] + cm[3, 3])
        # compute accuracy for each class
        accuracy_class_1 = cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[0, 2] + cm[0, 3] + cm[1, 0] + cm[2, 0] + cm[3, 0])
        accuracy_class_2 = cm[1, 1] / (cm[1, 0] + cm[1, 1] + cm[1, 2] + cm[1, 3] + cm[0, 1] + cm[2, 1] + cm[3, 1])
        accuracy_class_3 = cm[2, 2] / (cm[2, 0] + cm[2, 1] + cm[2, 2] + cm[2, 3] + cm[0, 2] + cm[1, 2] + cm[3, 2])
        accuracy_class_4 = cm[3, 3] / (cm[3, 0] + cm[3, 1] + cm[3, 2] + cm[3, 3] + cm[0, 3] + cm[1, 3] + cm[2, 3])
        # net recall
        average_class_accuracy = (accuracy_class_1 + accuracy_class_2 + accuracy_class_3 + accuracy_class_4) / 4.
        performance = [accuracy, average_class_accuracy, accuracy_class_1, accuracy_class_2, accuracy_class_3, accuracy_class_4]
        return performance

# This main is used to train the network and test if the functions work
if __name__ == '__main__':
    
    # CONSTANTS
    FILE_PATH = "dataset/image"
    FILE_PATH_LABELS = "dataset/label.csv"
    TEST_FILE_PATH = "test/image"
    TEST_FILE_PATH_LABELS = "test/label.csv"
    NUMBER_OF_FOLDS = 2 # 5
    BATCH_SIZE = 64
    NB_EPOCHS = 1 # 50
    VERBOSITY = 0
    IMAGE_DIMENSION_XY = 256
    
    ## parameter of the convolutional neural network
    # Convolutional Layer Arguments
    first_convolutional_layer_arguments = select_1st_convolutional_layer_arguments(64, (5, 5), 'Same', 'relu', (IMAGE_DIMENSION_XY, IMAGE_DIMENSION_XY, 3))
    second_convolutional_layer_arguments = select_2nd_convolutional_layer_arguments(128, (3, 3), 'Same', 'relu')
    third_convolutional_layer_arguments = select_2nd_convolutional_layer_arguments(128, (3, 3), 'Same', 'relu')
    fourth_convolutional_layer_arguments = select_2nd_convolutional_layer_arguments(128, (2, 2), 'Same', 'relu')
    fifth_convolutional_layer_arguments = select_2nd_convolutional_layer_arguments(256, (2, 2), 'Same', 'relu')
    # Pooling Layer Arguments
    first_pooling_layer_arguments = select_pooling_layer_arguments(4, 4)
    second_pooling_layer_arguments = select_pooling_layer_window_size_arguments(2, 2, 2, 2)
    third_pooling_layer_arguments = select_pooling_layer_window_size_arguments(2, 2, 2, 2)
    fourth_pooling_layer_arguments = select_pooling_layer_window_size_arguments(2, 2, 2, 2)
    fifth_pooling_layer_arguments = select_pooling_layer_window_size_arguments(2, 2, 2, 2)
    # Hidden Layer Arguments
    hidden_layer_argument1 = select_hidden_layer_arguments(1024)
    # hidden_layer_arg2 = select_hidden_layer_args(512)
    # Output Layer Argument
    output_layer_argument1 = 1
    output_layer_argument2 = 2
    output_layer_argument3 = 4
    DROPOUT_TRUE = True
    DROPOUT_FALSE = False
    
    # Preprocess the data
    CATs, OHE_labels, FILE_names_multiclass, labels, FILE_names_binary, BINARY_labels, BINARY_OHE_labels, one_hot_encoder_4_categories, one_hot_encoder_2_categories, FILE_names, BINARY_labels_OG, BINARY_OHE_labels_OG, OHE_labels_OG, LABEL_ENCODER, LABEL_ENCODER_BINARY = preprocess_data(FILE_PATH_LABELS, "smote")
    

    
    # Train 6 Deep CNN classifiers and apply cross validation
    
    # Binary Classifier with one output node and no dropout
    classifier_1_OUT_XDROP = train_and_crossvalidate(FILE_names_binary,
                                                      BINARY_labels,
                                                      FILE_PATH,
                                                      NUMBER_OF_FOLDS,
                                                      BATCH_SIZE,
                                                      NB_EPOCHS,
                                                      VERBOSITY,
                                                      IMAGE_DIMENSION_XY,
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
                                                      hidden_layer_argument1,
                                                      output_layer_argument1,
                                                      DROPOUT_FALSE)
    
    # Evaluate the performance on the original dataset
    performance_1_OUT_XDROP = test_and_evaluate(FILE_names,
                                                FILE_PATH,
                                                BINARY_labels_OG,
                                                classifier_1_OUT_XDROP,
                                                IMAGE_DIMENSION_XY,
                                                LABEL_ENCODER_BINARY)
    
    # save the model
    classifier_1_OUT_XDROP.saveCNNWeights('/Users/TEB/Documents/Angletterre/London/Cambridge-UCL/Modules/AppliedMachineLearningSystems_I/Assignment/classifier_1_OUT_XDROP1.h5')
    
    
    # Binary Classifier with one output node and no dropout
    classifier_2_OUT_XDROP = train_and_crossvalidate(FILE_names_binary,
                                                      BINARY_OHE_labels,
                                                      FILE_PATH,
                                                      NUMBER_OF_FOLDS,
                                                      BATCH_SIZE,
                                                      NB_EPOCHS,
                                                      VERBOSITY,
                                                      IMAGE_DIMENSION_XY,
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
                                                      hidden_layer_argument1,
                                                      output_layer_argument2,
                                                      DROPOUT_FALSE)
    
    performance_2_OUT_XDROP = test_and_evaluate(FILE_names,
                                                FILE_PATH,
                                                BINARY_OHE_labels_OG,
                                                classifier_2_OUT_XDROP,
                                                IMAGE_DIMENSION_XY,
                                                LABEL_ENCODER_BINARY,
                                                one_hot_encoder_2_categories)
    
    classifier_2_OUT_XDROP.saveCNNWeights('/Users/TEB/Documents/Angletterre/London/Cambridge-UCL/Modules/AppliedMachineLearningSystems_I/Assignment/classifier_2_OUT_XDROP1.h5')
    
    # Multi-class Classifier with 4 output nodes and no dropout
    classifier_4_OUT_XDROP = train_and_crossvalidate(FILE_names_multiclass,
                                                      OHE_labels,
                                                      FILE_PATH,
                                                      NUMBER_OF_FOLDS,
                                                      BATCH_SIZE,
                                                      NB_EPOCHS,
                                                      VERBOSITY,
                                                      IMAGE_DIMENSION_XY,
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
                                                      hidden_layer_argument1,
                                                      output_layer_argument3,
                                                      DROPOUT_FALSE)
    
    
    performance_4_OUT_XDROP = test_and_evaluate(FILE_names,
                                                FILE_PATH,
                                                OHE_labels_OG,
                                                classifier_4_OUT_XDROP,
                                                IMAGE_DIMENSION_XY,
                                                LABEL_ENCODER,
                                                one_hot_encoder_4_categories)
    
    classifier_4_OUT_XDROP.saveCNNWeights('/Users/TEB/Documents/Angletterre/London/Cambridge-UCL/Modules/AppliedMachineLearningSystems_I/Assignment/classifier_4_OUT_XDROP1.h5')
    
    # Binary Classifier with one output node and dropout
    classifier_1_OUT = train_and_crossvalidate(FILE_names_binary,
                                                BINARY_labels,
                                                FILE_PATH,
                                                NUMBER_OF_FOLDS,
                                                BATCH_SIZE,
                                                NB_EPOCHS,
                                                VERBOSITY,
                                                IMAGE_DIMENSION_XY,
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
                                                hidden_layer_argument1,
                                                output_layer_argument1,
                                                DROPOUT_TRUE)
    
    performance_1_OUT = test_and_evaluate(FILE_names,
                                          FILE_PATH,
                                          BINARY_labels_OG,
                                          classifier_1_OUT,
                                          IMAGE_DIMENSION_XY,
                                          LABEL_ENCODER_BINARY)
    
    classifier_1_OUT.saveCNNWeights('/Users/TEB/Documents/Angletterre/London/Cambridge-UCL/Modules/AppliedMachineLearningSystems_I/Assignment/classifier_1_OUT1.h5')
    
    # Binary Classifier with one output node and dropout
    classifier_2_OUT = train_and_crossvalidate(FILE_names_binary,
                                                BINARY_OHE_labels,
                                                FILE_PATH,
                                                NUMBER_OF_FOLDS,
                                                BATCH_SIZE,
                                                NB_EPOCHS,
                                                VERBOSITY,
                                                IMAGE_DIMENSION_XY,
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
                                                hidden_layer_argument1,
                                                output_layer_argument2,
                                                DROPOUT_TRUE)
    
    performance_2_OUT = test_and_evaluate(FILE_names,
                                          FILE_PATH,
                                          BINARY_OHE_labels_OG,
                                          classifier_2_OUT,
                                          IMAGE_DIMENSION_XY,
                                          LABEL_ENCODER_BINARY,
                                          one_hot_encoder_2_categories)
    
    classifier_2_OUT.saveCNNWeights('/Users/TEB/Documents/Angletterre/London/Cambridge-UCL/Modules/AppliedMachineLearningSystems_I/Assignment/classifier_2_OUT1.h5')
    
    
    # Multi-class Classifier with 4 output nodes and dropout
    classifier_4_OUT = train_and_crossvalidate(FILE_names_multiclass,
                                                OHE_labels,
                                                FILE_PATH,
                                                NUMBER_OF_FOLDS,
                                                BATCH_SIZE,
                                                NB_EPOCHS,
                                                VERBOSITY,
                                                IMAGE_DIMENSION_XY,
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
                                                hidden_layer_argument1,
                                                output_layer_argument3,
                                                DROPOUT_TRUE)
    
    performance_4_OUT = test_and_evaluate(FILE_names,
                                          FILE_PATH,
                                          OHE_labels_OG,
                                          classifier_4_OUT,
                                          IMAGE_DIMENSION_XY,
                                          LABEL_ENCODER,
                                          one_hot_encoder_4_categories)
    
    classifier_4_OUT.saveCNNWeights('/Users/TEB/Documents/Angletterre/London/Cambridge-UCL/Modules/AppliedMachineLearningSystems_I/Assignment/classifier_4_OUT1.h5')
    
    
    
    
    
    