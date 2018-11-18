#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tobiasbraun
"""
############################ imports ##########################################

import pickle
import random
import CLASS_NaiveBayesClassifier as naive

########################### testing ###########################################

#random.seed(1)

with open ("pickles/count_features/test_features","rb") as f:
    reviews_count_format=pickle.load(f)

with open ("pickles/test_scores","rb") as f:
    ratings=pickle.load(f)

n = 10 
m = 100
performance_threshold = 0.64
sum_performance = 0

for j in range(n):
    count = 0
    for i in range(m):
        x = random.randint(1,100000)
        word_to_be_predicted = reviews_count_format.tocsr()[x:x+1]
        NaiveTester=naive.NaiveBayesClassifier(reviews_count_format, ratings,
                                               word_to_be_predicted)
        if(NaiveTester.predict()==ratings[x:x+1]):
            count+=1
    #print("Predicted: " + str(NaiveTester.predict()) +"  " + "Actual: " +
    #     str(ratings[x:x+1]))
    percentage_performance = count/m
    #print(percentage_performance)
    sum_performance += percentage_performance

average_performance = sum_performance / n
#print(average_performance)
assert (average_performance >= performance_threshold)
