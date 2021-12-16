#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:22:06 2021

@author: TEB
"""

# import NumPy
import numpy as np

# Pandas
import pandas as pd

# scikit-learn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Import all Linear Classifiers
from LinearClassifiers import LinearClassifiers

# import the princpal component analysis
from PrincipleComponentAnalysis import PrincipleComponents


# imblearn imports
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE

# matplotlib
from matplotlib import pyplot as plt

# import python-openCV
import cv2

# Initial Preprocessing get the inputs and outputs
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
        

# Image Preprocessing
def get_image(file_path, img_file_name, img_size = 150):
    img = cv2.imread(file_path + "/" + img_file_name, 0)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.
    return img


# Preprocess all the images
def preprocess_input_data(input_file_names, file_path, target_image_dimension):
    
    # set an array for the inputs
    inputs = []
    
    # gather the data from individual pixels
    for i in range(0, len(input_file_names)):
        img = get_image(file_path, input_file_names[i],
                        target_image_dimension).flatten()
        inputs.append(img)
    
    # convert to numpy array
    inputs = np.array(inputs)
    
    return inputs


# train function for the linear classifiers
def train_and_crossvalidate(input_file_names,
                            file_path,
                            targets,
                            target_image_dimension_lr,
                            target_image_dimension_svc,
                            grid_parameters,
                            lr_nb_training_iterations,
                            lr_cost_sampling_rate,
                            lr_learning_rate,
                            lr_verbose,
                            nb_folds,
                            label_encoder = None,
                            nb_components_lr = 0,
                            nb_components_svc = 0,
                            pca_type_lr = "pca",
                            pca_type_svc = "pca"):
    
    # Tables to store the accuracies
    lr_accuracies_per_fold = []
    svcgs_accuracies_per_fold = []
    flat_image_dimension_lr = target_image_dimension_lr ** 2
    flat_image_dimension_svc = target_image_dimension_svc ** 2
    principle_components_lr = None
    principle_components_svc = None
    
    # set up the inputs and the outputs
    inputs_lr = preprocess_input_data(input_file_names,
                                      file_path,
                                      target_image_dimension_lr)
    inputs_svc = preprocess_input_data(input_file_names,
                                       file_path,
                                       target_image_dimension_svc)
    targets = targets
    
    # introduce principal component analysis if available
    if nb_components_lr > 0:
        principle_components_lr = PrincipleComponents(nb_components_lr,
                                                      pca_type_lr)
        principle_components_lr.fitPrincipleComponentAnalysis(inputs_lr)
        inputs_lr = principle_components_lr.augmentData(inputs_lr)
        flat_image_dimension_lr = nb_components_lr
    
    if nb_components_svc > 0:
        principle_components_svc = PrincipleComponents(nb_components_svc,
                                                       pca_type_svc)
        principle_components_svc.fitPrincipleComponentAnalysis(inputs_svc)
        inputs_svc = principle_components_svc.augmentData(inputs_svc)
        flat_image_dimension_svc = nb_components_svc
    
    # set up the best_classifier variable
    best_classifier_set = LinearClassifiers(flat_image_dimension_lr,
                                            flat_image_dimension_svc,
                                            grid_parameters,
                                            lr_nb_training_iterations,
                                            lr_cost_sampling_rate,
                                            lr_learning_rate,
                                            lr_verbose)
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits = nb_folds, shuffle = True)
    
    # K-fold Cross Validation model evaluation
    fold_number = 1
    folds = np.linspace(1, nb_folds, nb_folds)
    
    # Iterate over the k-folds Logistic Regressors
    for train, test in kfold.split(inputs_lr, targets):
        
        
        # Set up all the linear classifiers
        classifiers = LinearClassifiers(flat_image_dimension_lr,
                                        flat_image_dimension_svc,
                                        grid_parameters,
                                        lr_nb_training_iterations,
                                        lr_cost_sampling_rate,
                                        lr_learning_rate,
                                        lr_verbose)
        
        # Generate a print to indicate the number of the current k-fold
        print('------------------------------------------------------------------------')
        print(f'Training Logistic Regressor for fold {fold_number} ...')
        
        # prepare the data for the CNN
        X_train, X_val, Y_train, Y_val = train_test_split(inputs_lr[train],
                                                          targets[train],
                                                          test_size = 0.2,
                                                          random_state = 42)
        
        # Fit the data to the models
        classifiers.LR.trainLogisticRegressor(inputs_lr[train], targets[train])
        
        # Generate generalization metrics
        lr_test_accuracy, lr_test_recall, lr_test_specificity = classifiers.LR.evaluateLogisticRegressor(inputs_lr[test],
                                                                                                         targets[test])
        
        # Record the accuracies
        lr_accuracies_per_fold.append(lr_test_accuracy)
        
        # Increase fold number
        fold_number = fold_number + 1
        
        
        # Update the best parameters for Logistic Regressor and performance score
        if lr_test_accuracy * 100 > best_classifier_set.LRAccuracy * 100:
            best_classifier_set.LRAccuracy = lr_test_accuracy
            best_classifier_set.LR = classifiers.LR
            best_classifier_set.LRRecall = lr_test_recall
            best_classifier_set.LRSpecificity = lr_test_specificity
        
    # Print the current best test set accuracies
    print(f'Best Logistic Regressor Test Accuracy: {best_classifier_set.LRAccuracy * 100} %')
    print("END OF FOLD -----------------------------------****************---------------------------------")
    
    # Reset the flod number to 1
    fold_number = 1
    
    # Iterate over the k-folds Support Vector Classifier
    for train, test in kfold.split(inputs_svc, targets):
        
        # Set up all the linear classifiers
        classifiers = LinearClassifiers(flat_image_dimension_lr,
                                        flat_image_dimension_svc,
                                        grid_parameters,
                                        lr_nb_training_iterations,
                                        lr_cost_sampling_rate,
                                        lr_learning_rate,
                                        lr_verbose)
        
        # Generate a print to indicate the number of the current k-fold
        print('------------------------------------------------------------------------')
        print(f'Training Support Vector Classifier for fold {fold_number} ...')
        
        # Fit the data to the models
        classifiers.SVCGS.trainSupportVectorClassifier(inputs_svc[train],
                                                       targets[train])
        
        # Generate generalization metrics
        svcgs_accuracy, svcgs_recall, svcgs_specificity = classifiers.SVCGS.evaluateSupportVectorClassifier(inputs_svc[test],
                                                                                                            targets[test])
        
        # Record the accuracies
        svcgs_accuracies_per_fold.append(svcgs_accuracy)
        
        # Increase fold number
        fold_number = fold_number + 1
        
        
        # Update the best parameters for the Support Vector Classifier
        if svcgs_accuracy * 100 > best_classifier_set.SVCGSAccuracy * 100:
            best_classifier_set.SVCGSAccuracy = svcgs_accuracy
            best_classifier_set.SVCGS = classifiers.SVCGS
            best_classifier_set.SVCGSRecall = svcgs_recall
            best_classifier_set.SVCGSSpecificity = svcgs_specificity
        
        
        print(f'Best Support Vector Machine Classifier Test Accuracy: {best_classifier_set.SVCGSAccuracy * 100} %')
        print("END OF FOLD -----------------------------------****************---------------------------------")
    
    
    
    ## LR Accuracies
    
    # Train Accuracy over K-Folds
    plt.figure()
    plt.title("Logistic Regression Test Accuracy per Fold")
    plt.xlabel("Fold number")
    plt.ylabel("Accuracy in percentage [%]")
    plt.plot(folds, lr_accuracies_per_fold)
    
    # Train Accuracy over K-Folds
    plt.figure()
    plt.title("Support Vector Machine Classifier Test Accuracy per Fold")
    plt.xlabel("Fold number")
    plt.ylabel("Accuracy in percentage [%]")
    plt.plot(folds, svcgs_accuracies_per_fold)
    
    return best_classifier_set, principle_components_lr, principle_components_svc

## Test function for the Linear classifiers
def test_and_evaluate(input_file_names,
                      file_path,
                      labels,
                      models,
                      target_image_dimension_lr,
                      target_image_dimension_svc,
                      label_encoder = None,
                      pca_lr = None,
                      pca_svc = None):
    
    # set up the inputs and the outputs
    inputs_lr = preprocess_input_data(input_file_names,
                                      file_path,
                                      target_image_dimension_lr)
    inputs_svc = preprocess_input_data(input_file_names,
                                       file_path,
                                       target_image_dimension_svc)
    
    targets = labels
    
    # pca and lr
    if pca_lr is not None:
        inputs_lr = pca_lr.augmentData(inputs_lr)
    
    # pca and svc
    if pca_svc is not None:
        inputs_svc = pca_svc.augmentData(inputs_svc)
        
    
    # evaluate the linear classifiers
    models.LRAccuracy, models.LRRecall, models.LRSpecificity = models.LR.evaluateLogisticRegressor(inputs_lr,
                                                                                                   targets,
                                                                                                   label_encoder)
    models.SVCGSAccuracy, models.SVCGSRecall, models.SVCGSSpecificity = models.SVCGS.evaluateSupportVectorClassifier(inputs_svc,
                                                                                                                     targets,
                                                                                                                     label_encoder)
    
    # Get the performance
    performance = [models.LRAccuracy,
                   models.LRRecall,
                   models.LRSpecificity,
                   models.SVCGSAccuracy,
                   models.SVCGSRecall,
                   models.SVCGSSpecificity]
    
    return performance

# This main is used to train the network and test if the functions work
if __name__ == '__main__':
    
    # CONSTANTS
    FILE_PATH = "dataset/image"
    FILE_PATH_LABELS = "dataset/label.csv"
    TEST_FILE_PATH = "test/image"
    TEST_FILE_PATH_LABELS = "test/label.csv"
    NUMBER_OF_FOLDS = 5
    LOGISTIC_REGRESSOR_VERBOSE = 0
    IMAGE_DIMENSIONS_XY_LR = 128
    IMAGE_DIMENSIONS_XY_SVM = 16
    GRID_PARAMETERS = {'C':[0.1, 1, 10, 100],
                       'gamma':[0.0001, 0.001, 0.1, 1],
                       'kernel':['rbf']}
    LOGISTIC_REGRESSOR_NUMBER_OF_TRAINING_ITERATIONS = 3000
    LOGISITIC_REGRESSOR_COST_SAMPLING_RATE = 10
    LOGISTIC_REGRESSOR_LEARNING_RATE = 0.3
    NUMBER_OF_PRINCIPLE_COMPONENTS_LR = 2048
    NUMBER_OF_PRINCIPLE_COMPONENTS_SVC = 256
    PCA_TYPE_LR = "pca"
    PCA_TYPE_SVC = "cpca"
    
    ## Parameters of the linear classifers
    # Convolutional Layer Arguments
    
    # Preprocess the data
    CATs, OHE_labels, FILE_names_multiclass, labels, FILE_names_binary, BINARY_labels, BINARY_OHE_labels, one_hot_encoder_4_categories, one_hot_encoder_2_categories, FILE_names, BINARY_labels_OG, BINARY_OHE_labels_OG, OHE_labels_OG, LABEL_ENCODER, LABEL_ENCODER_BINARY = preprocess_data(FILE_PATH_LABELS, "smote")
    TEST_CATs, TEST_OHE_labels, TEST_FILE_names_multiclass, TEST_labels, TEST_FILE_names_binary, TEST_BINARY_labels, TEST_BINARY_OHE_labels, TEST_one_hot_encoder_4_categories, TEST_one_hot_encoder_2_categories, TEST_FILE_names, TEST_BINARY_labels_OG, TEST_BINARY_OHE_labels_OG, TEST_OHE_labels_OG, TEST_LABEL_ENCODER, TEST_LABEL_ENCODER_BINARY = preprocess_data(TEST_FILE_PATH_LABELS, "smote")
    
    ## Train Linear Classifiers
    classifier_set, _, _ = train_and_crossvalidate(FILE_names_binary,
                                                   FILE_PATH,
                                                   BINARY_labels,
                                                   IMAGE_DIMENSIONS_XY_LR,
                                                   IMAGE_DIMENSIONS_XY_SVM,
                                                   GRID_PARAMETERS,
                                                   LOGISTIC_REGRESSOR_NUMBER_OF_TRAINING_ITERATIONS,
                                                   LOGISITIC_REGRESSOR_COST_SAMPLING_RATE,
                                                   LOGISTIC_REGRESSOR_LEARNING_RATE,
                                                   LOGISTIC_REGRESSOR_VERBOSE,
                                                   NUMBER_OF_FOLDS,
                                                   LABEL_ENCODER_BINARY)
    
    # Test the Linear Classifiers
    performance_LR_SVM = test_and_evaluate(TEST_FILE_names,
                                           TEST_FILE_PATH,
                                           TEST_BINARY_labels_OG,
                                           classifier_set,
                                           IMAGE_DIMENSIONS_XY_LR,
                                           IMAGE_DIMENSIONS_XY_SVM,
                                           TEST_LABEL_ENCODER_BINARY)
    
    # !!! Note best grid search for SVM alone chose: {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'} !!!
    
    # # set the image size
    IMAGE_DIMENSIONS_XY_SVM = 128
    
    ## Train Linear Classifiers and classic PCA
    classifier_set_PCA, pca_lr, pca_svc = train_and_crossvalidate(FILE_names_binary,
                                                                  FILE_PATH,
                                                                  BINARY_labels,
                                                                  IMAGE_DIMENSIONS_XY_LR,
                                                                  IMAGE_DIMENSIONS_XY_SVM,
                                                                  GRID_PARAMETERS,
                                                                  LOGISTIC_REGRESSOR_NUMBER_OF_TRAINING_ITERATIONS,
                                                                  LOGISITIC_REGRESSOR_COST_SAMPLING_RATE,
                                                                  LOGISTIC_REGRESSOR_LEARNING_RATE,
                                                                  LOGISTIC_REGRESSOR_VERBOSE,
                                                                  NUMBER_OF_FOLDS,
                                                                  LABEL_ENCODER_BINARY,
                                                                  NUMBER_OF_PRINCIPLE_COMPONENTS_LR,
                                                                  NUMBER_OF_PRINCIPLE_COMPONENTS_SVC)
    
    
    # evaluate PCA combined with linear models
    performance_PCA_LR_SVM = test_and_evaluate(TEST_FILE_names,
                                               TEST_FILE_PATH,
                                               TEST_BINARY_labels_OG,
                                               classifier_set_PCA,
                                               IMAGE_DIMENSIONS_XY_LR,
                                               IMAGE_DIMENSIONS_XY_SVM,
                                               TEST_LABEL_ENCODER_BINARY,
                                               pca_lr,
                                               pca_svc)
    
    
    
     # !!! Note best grid search for SVM PCA chose: {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'} !!!
    
    
    ## Train Linear Classifiers and kernel PCA
    classifier_set_KPCA, kpca_lr, kpca_svc  = train_and_crossvalidate(FILE_names_binary,
                                                                      FILE_PATH,
                                                                      BINARY_labels,
                                                                      IMAGE_DIMENSIONS_XY_LR,
                                                                      IMAGE_DIMENSIONS_XY_SVM,
                                                                      GRID_PARAMETERS,
                                                                      LOGISTIC_REGRESSOR_NUMBER_OF_TRAINING_ITERATIONS,
                                                                      LOGISITIC_REGRESSOR_COST_SAMPLING_RATE,
                                                                      LOGISTIC_REGRESSOR_LEARNING_RATE,
                                                                      LOGISTIC_REGRESSOR_VERBOSE,
                                                                      NUMBER_OF_FOLDS,
                                                                      LABEL_ENCODER_BINARY,
                                                                      NUMBER_OF_PRINCIPLE_COMPONENTS_LR,
                                                                      NUMBER_OF_PRINCIPLE_COMPONENTS_SVC,
                                                                      PCA_TYPE_LR,
                                                                      PCA_TYPE_SVC)
    
    # evaluate kernel PCA and linear models
    performance_KPCA_LR_SVM = test_and_evaluate(TEST_FILE_names,
                                                TEST_FILE_PATH,
                                                TEST_BINARY_labels_OG,
                                                classifier_set_KPCA,
                                                IMAGE_DIMENSIONS_XY_LR,
                                                IMAGE_DIMENSIONS_XY_SVM,
                                                TEST_LABEL_ENCODER_BINARY,
                                                kpca_lr,
                                                kpca_svc)
    
    # !!! Note best grid search for SVM CPCA chose: {'C': 1, 'gamma': 1, 'kernel': 'rbf'} !!!
    
    
    
    
    
    
    