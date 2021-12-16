#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 15:24:34 2021

@author: TEB
"""

# Import tools
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA



class PrincipleComponents:
    
    def __init__(self, number_of_components, pca_type = "pca"):
        
        if pca_type == "pca":
            self.PrincipleComponentAnalysis = PCA(number_of_components)
        
        if pca_type == "kpca":
            self.PrincipleComponentAnalysis = KernelPCA(number_of_components,
                                                        kernel = "rbf")
        
        if pca_type == "ppca":
            self.PrincipleComponentAnalysis = KernelPCA(number_of_components,
                                                        kernel = "poly")
            
        if pca_type == "spca":
            self.PrincipleComponentAnalysis = KernelPCA(number_of_components,
                                                        kernel = "sigmoid")
        
        if pca_type == "cpca":
            self.PrincipleComponentAnalysis = KernelPCA(number_of_components,
                                                        kernel = "cosine")
    
    def fitPrincipleComponentAnalysis(self, inputs):
        self.PrincipleComponentAnalysis.fit(inputs)
    
    def augmentData(self, inputs):
        PrincipleComponents = self.PrincipleComponentAnalysis.transform(inputs)
        return PrincipleComponents