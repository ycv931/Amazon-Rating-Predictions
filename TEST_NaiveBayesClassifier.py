#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tobiasbraun
"""
############################ imports ##########################################

import pickle
import random
import CLASS_NaiveBayesClassifier as naive
import numpy as np

########################### testing ###########################################

#random.seed(1)

with open ("pickles/count_features/test_features","rb") as f:
    reviews_count_format=pickle.load(f)

with open ("pickles/test_scores","rb") as f:
    ratings=pickle.load(f)

n = 10
m = 1000
performance_threshold = 0.64
sum_performance = 0

naiveTester = naive.NaiveBayesClassifier()
naiveTester.train(reviews_count_format, ratings)
for j in range(n):
    count = 0
    x = random.randint(1,100000)
    words_to_be_predicted = reviews_count_format.tocsr()[x:x+m]
    real_ratings = ratings[x:x+m]
    diff = naiveTester.fit(words_to_be_predicted) - real_ratings
    count = m - np.count_nonzero (diff)
    percentage_performance = count/m
    sum_performance += percentage_performance
    
    #print("Predicted:     " + str(naiveTester.fit(words_to_be_predicted)))
    #print("Actual rating: " + str(real_ratings))
average_performance = sum_performance / n
#print(average_performance)
assert (average_performance >= performance_threshold)

