#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tobiasbraun
"""

import numpy as np
import scipy.sparse as spa

class NaiveBayesClassifier():    
    def __init__(self, rev: spa.csr_matrix, 
                 scores: np.ndarray, x: spa.csr_matrix):
        self.rev = rev
        self.scores = scores
        self.to_predict = x
            
    def get_fractions(self):
        """assigns the fraction of all reviews belonging to each rating value"""
        count = np.bincount(self.scores)[1:]
        fractions = count / np.sum(count)
        
        return fractions
        
    def convert_data(self):
        """creates a list in which each entry represents a sparse matrices 
        with the sum of all the words of all the reviews that have the same 
        rating."""
        rev_to_rating_list = [np.where(self.scores == i) for i in range (1,6)]
        rev_to_rating = [self.rev.tocsr()[rev_to_rating_list[i]]
        for i in range(5)]
        rev_to_rating_sum = [rev_to_rating[i].sum(axis = 0) for i in range(5)]
        
        return rev_to_rating_sum
    
    def calculate_conditional_prob(self):
        """calculates the conditional probability of a word appearing in each 
        rating category. Includes Laplace Smoothing with low ɑ."""
        a=0.001
        size = self.rev.get_shape()[1]
        temp = self.convert_data()
        cond_prob = [(temp[i]+a)/(temp[i].sum()+1+size) for i in range(5)]
        
        return cond_prob
                
    def predict(self):
        """returns a list with the probabilities of x having a certain 
        (index number + 1)rating. Applies logs to avoid underflow and
        to take into account that the probability of appearance of a word
        increases if the same word has already appeared before."""
        
        # To improve performance increasing the weight of non-5 star rating
        # concordance (improvement neglible ~ < 1%)
        # count = np.bincount(self.scores)[1:]
        # new_weights = [(count[4] / count[i]) for i in range(5)]
        #
        cond_prob_temp = self.calculate_conditional_prob()
        x = self.to_predict
        x_indices = spa.find(x)
        # x_indices breaks the sparse matrix down to the few values containing
        # a number different from 0
        predictions = [0]*5
        for i in range(5):
            for j in range(0,len(x_indices[0])):
                predictions[i] += self.get_fractions()[i] * (
                        ((cond_prob_temp[i][x_indices[0][j],
                        x_indices[1][j]]))** np.log(1 + x_indices[2][j]))
        smoothed_predictions = [np.log(predictions[i]) if (predictions[i]!=0) 
                                else float("-inf") for i in range(5)]
        final_predict = smoothed_predictions.index(max(smoothed_predictions)) + 1
        
        return final_predict             
