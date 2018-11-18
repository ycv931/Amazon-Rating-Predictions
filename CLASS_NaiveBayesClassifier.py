#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tobiasbraun
"""

import numpy as np
import scipy.sparse as spa

class NaiveBayesClassifier():    
    
    def __init__(self):
        self.is_trained = False
    
    def train(self, train_data: spa.csr_matrix, scores: np.ndarray, alpha = 0.001):
        """trains the classifier."""
        self.is_trained = True
        self.train_data = train_data
        self.scores = scores
        self.fractions = self.get_fractions()
        self.rev_to_rating_sum = self.convert_data()
        self.cond_prob = self.calculate_conditional_prob(alpha)
        
        
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
        rev_to_rating = [self.train_data.tocsr()[rev_to_rating_list[i]]
        for i in range(5)]
        rev_to_rating_sum = [rev_to_rating[i].sum(axis = 0) for i in range(5)]
        
        return rev_to_rating_sum
    
    def calculate_conditional_prob(self, a = 0.001):
        """calculates the conditional probability of a word appearing in each 
        rating category. Includes Laplace Smoothing with low ɑ."""
        size = self.train_data.get_shape()[1]
        temp = self.rev_to_rating_sum
        self.cond_prob = [(temp[i]+a)/(temp[i].sum()+1+size) for i in range(5)]
        
        return self.cond_prob
                
    def fit(self, x: spa.csr_matrix):
        """returns a numpy.array with the predicted ratings. Applies logs to 
        avoid underflow and to take into account that the probability of 
        appearance of a word increases if the same word has already appeared before."""
       
        if self.is_trained == False:
            return ("""The Classifier has not been trained. Please use 
                    train(train_data: spa.csr_matrix, scores: np.ndarray, 
                    Laplace_alpha) to train the Classifier.""")
        else:
            final_predict = np.empty(0, dtype = int)
            for row in x:
                row_indices = spa.find(row)
                predictions = [0]*5
                for i in range(5):
                    for j in range(0,len(row_indices[0])):
                        predictions[i] += self.fractions[i] * (
                        ((self.cond_prob[i][row_indices[0][j],
                        row_indices[1][j]])) ** np.log(1 + row_indices[2][j]))
                smoothed_predictions = [np.log(predictions[i]) if (predictions[i] != 0) 
                                    else float("-inf") for i in range(5)]
                final_predict = np.append(final_predict, 
                smoothed_predictions.index(max(smoothed_predictions)) + 1)
                
        return final_predict             
