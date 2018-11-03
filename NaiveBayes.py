import numpy as np

#test set
#numbers represent the count of words "good","great","love","bad","hate"
vocabvectorlength=5
numberofreviews=6 
ratingScale=["1","2","3","4","5"]
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

#Calculate the probability of each word per class

# word i appears k times in class j:

wordcount=[0]*len(x0)
for i in range(0,len(x0)):
    for word in X1:
        wordcount[i]+=word[i]
wordcount
    
# calculate wordCountPerClass

wordCountPerClass={"1": np.array((0.0,0.0,0.0,0.0,0.0)),
                    "2": np.array((0.0,0.0,0.0,0.0,0.0)),
                    "3": np.array((0.0,0.0,0.0,0.0,0.0)),
                    "4": np.array((0.0,0.0,0.0,0.0,0.0)),
                    "5": np.array((0.0,0.0,0.0,0.0,0.0))}
for rev in range (0,len(X1)):
    wordCountPerClass[str(Y[rev])]+=X1[rev]


#calculate P(word i \| class j )
alpha=0.00001
for rating in wordCountPerClass.keys():
    for i in range(0,vocabvectorlength):
        temp=wordCountPerClass[rating]
        a=wordCountPerClass[rating][i]+alpha
        b=(sum(wordCountPerClass[rating])+vocabvectorlength+1)
        temp[i]=a/b


#calculating the probability of class j
def MNB(revue):
    """returns a list with the probability of 
    a revue belonging to a certain class"""
    probabilityOfj=np.array((0.0,0.0,0.0,0.0,0.0))
    for rating in pi:    
        temp=wordCountPerClass[str(rating)]
        result=1
        for word in range(0,len(revue)):
            result=result*(temp[word]**(revue[word]))
        
        
        classValue=rating * result
        probabilityOfj[rating-1]=classValue
    
    maxLikelyClass=np.argmax(probabilityOfj)
    return("Rating prediction of " + str(revue)+" : " + ratingScale[maxLikelyClass])


print(MNB([1,0,0,0,0]))
print(MNB([0,0,0,5,4]))
print(MNB([0,0,4,2,0]))
print(MNB([1,1,1,1,1]))
print(MNB([1,0,0,0,0]))
print(MNB([5,4,2,3,0]))
    

