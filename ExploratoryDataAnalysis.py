#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:22:06 2021

@author: TEB
"""

# import Numpy
import numpy as np

# matplotlib
from matplotlib import pyplot as plt

# TrainCNN imports
from TrainCNN import preprocess_data

def count_binary_data(binary_data):
    
    # set up the counts
    histogram_counts = [0, 0]
    # turn the array into a list
    binary_data = list(binary_data.flatten())
    
    # Increment the different categories -> 0 no_tumor, 1 tumor
    for i in range(0, len(binary_data)):    
        if binary_data[i] == "no_tumor":
            histogram_counts[0] += 1
        else:
            histogram_counts[1] += 1
    
    return histogram_counts

def count_labels_data(data):
    
    # set up the counts
    histogram_counts = [0, 0, 0, 0]
    # turn the array into a list
    data = list(data.flatten())
    
    # Increment the different categories -> 0 no_tumor, 1 tumor, 2, 3
    for i in range(0, len(data)):
        if data[i] == "no_tumor":
            histogram_counts[0] += 1
        elif data[i] == "meningioma_tumor":
            histogram_counts[1] += 1
        elif data[i] == "glioma_tumor":
            histogram_counts[2] += 1
        elif data[i] == "pituitary_tumor":
            histogram_counts[3] += 1
    
    return histogram_counts

def plot_histograms(labels_og_counts, labels_counts, x_labels, title):
    
    # Set up the x-axis
    X_axis = np.arange(len(x_labels))
    
    # Plot the Data distributions
    plt.figure()
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Counts")
    plt.xticks(X_axis, x_labels)
    plt.bar(X_axis - 0.2, labels_og_counts, 0.2, label = "original")
    plt.bar(X_axis + 0.2, labels_counts, 0.2, label = "resampled")
    plt.legend()
    return

def plot_histogram_original(labels_og_counts, x_labels, title):
    
    # Set up the x-axis
    X_axis = np.arange(len(x_labels))
    
    # Plot the Data distributions
    plt.figure()
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Counts")
    plt.xticks(X_axis, x_labels)
    plt.bar(X_axis, labels_og_counts, 0.2)
    return

def exploratory_data_analysis(binary_labels_og, binary_labels, ohe_labels, ohe_labels_og, one_hot_encoder_4_categories, label_encoder, label_encoder_binary):
    
    # Recover binary labels original
    binary_labels_og = label_encoder_binary.inverse_transform(binary_labels_og)
    
    # Recover binary labels (after smote)
    binary_labels = label_encoder_binary.inverse_transform(binary_labels)
    
    # Recover original labels
    labels_og = one_hot_encoder_4_categories.inverse_transform(ohe_labels_og)
    labels_og = label_encoder.inverse_transform(labels_og)
    
    # Recover labels after smote
    labels = one_hot_encoder_4_categories.inverse_transform(ohe_labels)
    labels = label_encoder.inverse_transform(labels)
    
    # Binary histogram preprocessing 
    binary_labels_og_counts = count_binary_data(binary_labels_og)
    binary_labels_counts = count_binary_data(binary_labels)
    
    # Labels histogram preprocessing
    labels_og_counts = count_labels_data(labels_og)
    labels_counts = count_labels_data(labels)
    
    # Plot binary label histogram
    binary_x_labels = ["No Tumor", "Tumor"]
    title_binary = "Binary Label Distribution"
    plot_histograms(binary_labels_og_counts, binary_labels_counts, binary_x_labels, title_binary)
    
    # Plot labels histogram
    labels_x_labels = ["No Tumor", "Meningioma", "Glioma", "Pituitary Tumor"]
    title_labels = "Label Distribution"
    plot_histograms(labels_og_counts, labels_counts, labels_x_labels, title_labels)
    return

def explore_test_set(binary_labels_og, ohe_labels_og, one_hot_encoder_2_categories, one_hot_encoder_4_categories, label_encoder, label_encoder_binary):
    
    # Recover original labels
    labels_og = one_hot_encoder_4_categories.inverse_transform(ohe_labels_og)
    labels_og = label_encoder.inverse_transform(labels_og)
    
    # Recover the binary label original
    binary_labels_og = label_encoder_binary.inverse_transform(binary_labels_og)
    
    # Binary histogram preprocessing 
    binary_labels_og_counts = count_binary_data(binary_labels_og)
    
    # Labels histogram preprocessing
    labels_og_counts = count_labels_data(labels_og)
    
    # Plot binary original label histogram
    binary_x_labels = ["No Tumor", "Tumor"]
    title_binary = "Binary Label Distribution"
    plot_histogram_original(binary_labels_og_counts, binary_x_labels, title_binary)
    
    # Plot labels histogram
    labels_x_labels = ["No Tumor", "Meningioma", "Glioma", "Pituitary Tumor"]
    title_labels = "Label Distribution"
    plot_histogram_original(labels_og_counts, labels_x_labels, title_labels)

# This main is used to perform exploratory data analysis
if __name__ == '__main__':
    
    # CONSTANTS
    FILE_PATH = "dataset/image"
    FILE_PATH_LABELS = "dataset/label.csv"
    TEST_FILE_PATH = "test/image"
    TEST_FILE_PATH_LABELS = "test/label.csv"
    
    # Preprocess the data
    CATs, OHE_labels, FILE_names_multiclass, labels, FILE_names_binary, BINARY_labels, BINARY_OHE_labels, one_hot_encoder_4_categories, one_hot_encoder_2_categories, FILE_names, BINARY_labels_OG, BINARY_OHE_labels_OG, OHE_labels_OG, LABEL_ENCODER, LABEL_ENCODER_BINARY = preprocess_data(FILE_PATH_LABELS, "smote")
    TEST_CATs, TEST_OHE_labels, TEST_FILE_names_multiclass, TEST_labels, TEST_FILE_names_binary, TEST_BINARY_labels, TEST_BINARY_OHE_labels, TEST_one_hot_encoder_4_categories, TEST_one_hot_encoder_2_categories, TEST_FILE_names, TEST_BINARY_labels_OG, TEST_BINARY_OHE_labels_OG, TEST_OHE_labels_OG, TEST_LABEL_ENCODER, TEST_LABEL_ENCODER_BINARY = preprocess_data(TEST_FILE_PATH_LABELS, "smote")
    
    # Exploratory data analysis of the training set and resampled sets with SMOTE
    exploratory_data_analysis(BINARY_labels_OG,
                              BINARY_labels,
                              OHE_labels,
                              OHE_labels_OG,
                              one_hot_encoder_4_categories,
                              LABEL_ENCODER,
                              LABEL_ENCODER_BINARY)
    
    # Exploratory data analysis of the test set
    explore_test_set(TEST_BINARY_labels_OG,
                     TEST_OHE_labels_OG,
                     TEST_one_hot_encoder_2_categories,
                     TEST_one_hot_encoder_4_categories,
                     TEST_LABEL_ENCODER,
                     TEST_LABEL_ENCODER_BINARY)
    
    