#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 13:17:52 2021

@author: TEB
"""





# CNN_crossvalidation imports
from TrainCNN import preprocess_data
from TrainCNN import test_and_evaluate
from TrainCNN import CNN_Classifier
from TrainCNN import select_1st_convolutional_layer_arguments
from TrainCNN import select_2nd_convolutional_layer_arguments
from TrainCNN import select_pooling_layer_arguments
from TrainCNN import select_pooling_layer_window_size_arguments
from TrainCNN import select_hidden_layer_arguments


# initialize the model paths
classifier_1_OUT_XDROP_path = 'classifier_1_OUT_XDROP.h5'
classifier_1_OUT_path = 'classifier_1_OUT.h5'
classifier_2_OUT_XDROP_path = 'classifier_2_OUT_XDROP.h5'
classifier_2_OUT_path = 'classifier_2_OUT.h5'
classifier_2_OUT_100_path = 'classifier_2_OUT_100.h5'
classifier_4_OUT_XDROP_path = 'classifier_4_OUT_XDROP.h5'
classifier_4_OUT_path = 'classifier_4_OUT.h5'




if __name__ == "__main__":
    
    print("HELLO WORLD!")
    
    # CONSTANTS
    FILE_PATH = "dataset/image"
    FILE_PATH_LABELS = "dataset/label.csv"
    NUMBER_OF_FOLDS = 5
    BATCH_SIZE = 100
    NB_EPOCHS = 50
    VERBOSITY = 0
    IMAGE_DIMENSION_XY = 256
    
    
    
    # CONSTANTS
    FILE_PATH = "test/image"
    FILE_PATH_LABELS = "test/label.csv"
    NUMBER_OF_FOLDS = 2 # 5
    BATCH_SIZE = 64
    NB_EPOCHS = 1 # 50
    VERBOSITY = 0
    IMAGE_DIMENSION_XY = 256
    
    ## parameter of the convolutional neural networks
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
    hidden_layer_argument = select_hidden_layer_arguments(1024)
    # Output Layer Argument
    output_layer_argument1 = 1
    output_layer_argument2 = 2
    output_layer_argument3 = 4
    DROPOUT_TRUE = True
    DROPOUT_FALSE = False
    
    
    # Preprocess the data
    CATs, OHE_labels, FILE_names_multiclass, labels, FILE_names_binary, BINARY_labels, BINARY_OHE_labels, one_hot_encoder_4_categories, one_hot_encoder_2_categories, FILE_names, BINARY_labels_OG, BINARY_OHE_labels_OG, OHE_labels_OG, LABEL_ENCODER, LABEL_ENCODER_BINARY = preprocess_data(FILE_PATH_LABELS, "smote")
    
    ## classifier_1_OUT_XDROP
    # initialize the model
    classifier_1_OUT_XDROP = CNN_Classifier(first_convolutional_layer_arguments,
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
                                            output_layer_argument1,
                                            DROPOUT_FALSE)
    # load model weights
    classifier_1_OUT_XDROP.loadCNNWeights(classifier_1_OUT_XDROP_path)
    # test the model performance
    performance_1_OUT_XDROP = test_and_evaluate(FILE_names,
                                                FILE_PATH,
                                                BINARY_labels_OG,
                                                classifier_1_OUT_XDROP,
                                                IMAGE_DIMENSION_XY,
                                                LABEL_ENCODER_BINARY)
    
    
    
    
    ## classifier_1_OUT
    # initialize the model
    classifier_1_OUT = CNN_Classifier(first_convolutional_layer_arguments,
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
                                      output_layer_argument1,
                                      DROPOUT_TRUE)
    # load the model
    classifier_1_OUT.loadCNNWeights(classifier_1_OUT_path)
    # test the model performance
    performance_1_OUT = test_and_evaluate(FILE_names,
                                          FILE_PATH,
                                          BINARY_labels_OG,
                                          classifier_1_OUT,
                                          IMAGE_DIMENSION_XY,
                                          LABEL_ENCODER_BINARY)
    
    
    ## classifier_2_OUT_XDROP
    # initialize the model
    classifier_2_OUT_XDROP = CNN_Classifier(first_convolutional_layer_arguments,
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
                                            output_layer_argument2,
                                            DROPOUT_FALSE)
    # load model weights
    classifier_2_OUT_XDROP.loadCNNWeights(classifier_2_OUT_XDROP_path)
    # test the model performance
    performance_2_OUT_XDROP = test_and_evaluate(FILE_names,
                                                FILE_PATH,
                                                BINARY_OHE_labels_OG,
                                                classifier_2_OUT_XDROP,
                                                IMAGE_DIMENSION_XY,
                                                LABEL_ENCODER_BINARY,
                                                one_hot_encoder_2_categories)
    
    
    
    
    ## classifier_2_OUT
    # initialize the model
    classifier_2_OUT = CNN_Classifier(first_convolutional_layer_arguments,
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
                                      output_layer_argument2,
                                      DROPOUT_TRUE)
    # load the model weights
    classifier_2_OUT.loadCNNWeights(classifier_2_OUT_path)
    # test the model performance
    performance_2_OUT = test_and_evaluate(FILE_names,
                                          FILE_PATH,
                                          BINARY_OHE_labels_OG,
                                          classifier_2_OUT,
                                          IMAGE_DIMENSION_XY,
                                          LABEL_ENCODER_BINARY,
                                          one_hot_encoder_2_categories)
    
    
    ## classifier_2_OUT_100
    # initialize the model
    classifier_2_OUT_100 = CNN_Classifier(first_convolutional_layer_arguments,
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
                                          output_layer_argument2,
                                          DROPOUT_TRUE)
    # load the model weights
    classifier_2_OUT_100.loadCNNWeights(classifier_2_OUT_100_path)
    # test the model performance
    performance_2_OUT_100 = test_and_evaluate(FILE_names,
                                              FILE_PATH,
                                              BINARY_OHE_labels_OG,
                                              classifier_2_OUT_100,
                                              IMAGE_DIMENSION_XY,
                                              LABEL_ENCODER_BINARY,
                                              one_hot_encoder_2_categories)
    
    
    
    ## classifier_4_OUT_XDROP
    # initialize the model
    classifier_4_OUT_XDROP = CNN_Classifier(first_convolutional_layer_arguments,
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
                                            output_layer_argument3,
                                            DROPOUT_FALSE)
    # load model weights
    classifier_4_OUT_XDROP.loadCNNWeights(classifier_4_OUT_XDROP_path)
    # test the model performance
    performance_4_OUT_XDROP = test_and_evaluate(FILE_names,
                                                FILE_PATH,
                                                OHE_labels_OG,
                                                classifier_4_OUT_XDROP,
                                                IMAGE_DIMENSION_XY,
                                                LABEL_ENCODER,
                                                one_hot_encoder_4_categories)
    
    
    
    
    
    ## classifier_4_OUT
    # initialize the model
    classifier_4_OUT = CNN_Classifier(first_convolutional_layer_arguments,
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
                                      output_layer_argument3,
                                      DROPOUT_TRUE)
    # load the model weights
    classifier_4_OUT.loadCNNWeights(classifier_4_OUT_path)
    # test the model performance
    performance_4_OUT = test_and_evaluate(FILE_names,
                                          FILE_PATH,
                                          OHE_labels_OG,
                                          classifier_4_OUT,
                                          IMAGE_DIMENSION_XY,
                                          LABEL_ENCODER,
                                          one_hot_encoder_4_categories)
    
    
    