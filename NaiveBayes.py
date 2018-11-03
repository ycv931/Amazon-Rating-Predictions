import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

#test set
#numbers represent the count of words "good","great","love","bad","hate"
vocabvectorlength=5
numberofreviews=6 
x0=np.array((2,3,1,1,0))
X1=np.array(((2,2,2,0,0),(1,4,2,1,0),(0,1,0,2,2),
            (1,3,3,1,1),(0,0,0,1,1),(0,0,0,0,0)))
Y=np.array((4,5,1,3,1,3))

#pi contains the fractions of reviews with 1\, 2\,3\,4\,5 stars
pi={}

for i in range(1,len(Y)):
    count=0
    for j in range(0,len(Y)):
        if (Y[j]==i):
            count += 1
    pi[i]=count/len(Y)
pi

#Calculate the probability of each word per class

# word i appears k times in class j:

wordcount=[0]*len(x0)
for i in range(0,len(x0)):
    for word in X1:
        wordcount[i]+=word[i]
wordcount
    
# calculate wordCountPerClass

wordCountPerClass=[None]*len(set(Y))

for index in range(1,len(Y)):

    wordCountPerClass[Y[index]-1]+=X1[index]

wordCountPerClass