#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:48:42 2022

@author: michalmalek
"""

import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as metrics

from mlxtend.frequent_patterns import (apriori, association_rules)
from mlxtend.preprocessing import TransactionEncoder

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets

import numpy
import pandas 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

"""
inData = pandas.read_csv('1000Values.csv')
Y = inData['Value']
print(Y.describe())

plt.hist(Y)
plt.show()

def calcCD (Y, delta):
   maxY = numpy.max(Y)
   minY = numpy.min(Y)
   meanY = numpy.mean(Y)

   # Round the mean to integral multiples of delta
   middleY = delta * numpy.round(meanY / delta)

   # Determine the number of bins on both sides of the rounded mean
   nBinRight = numpy.ceil((maxY - middleY) / delta)
   nBinLeft = numpy.ceil((middleY - minY) / delta)
   lowY = middleY - nBinLeft * delta

   # Assign observations to bins starting from 0
   m = nBinLeft + nBinRight
   BIN_INDEX = 0;
   boundaryY = lowY
   for iBin in numpy.arange(m):
      boundaryY = boundaryY + delta
      BIN_INDEX = numpy.where(Y > boundaryY, iBin+1, BIN_INDEX)

   # Count the number of observations in each bins
   uBin, binFreq = numpy.unique(BIN_INDEX, return_counts = True)

   # Calculate the average frequency
   meanBinFreq = numpy.sum(binFreq) / m
   ssDevBinFreq = numpy.sum((binFreq - meanBinFreq)**2) / m
   CDelta = (2.0 * meanBinFreq - ssDevBinFreq) / (delta * delta)
   return(m, middleY, lowY, CDelta)

result = []
deltaList = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50]

for d in deltaList:
   nBin, middleY, lowY, CDelta = calcCD(Y,d)
   highY = lowY + nBin * d
   result.append([d, CDelta, lowY, middleY, highY, nBin])

   binMid = lowY + 0.5 * d + numpy.arange(nBin) * d
   plt.hist(Y, bins = binMid, align='mid')
   plt.title('Delta = ' + str(d))
   plt.ylabel('Number of Observations')
   plt.grid(axis = 'y')
   plt.show()

result = pandas.DataFrame(result, columns = {0:'Delta', 1:'C(Delta)', 2:'Low Y', 3:'Middle Y', 4:'High Y', 5:'N Bin'})

fig1, ax1 = plt.subplots()
ax1.set_title('Box Plot')
ax1.boxplot(Y, labels = ['Y'])
ax1.grid(linestyle = '--', linewidth = 1)
plt.show()




###################

train = {"A":[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],"Y":[0,0,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,1,0,1,1,1,0,1,1]}
train = pandas.DataFrame(train,columns = ["A","Y"])
train.head()

X = numpy.array(train["A"])
Y = numpy.array(train["Y"])

X = numpy.reshape(X,(25,1))
Y = numpy.reshape(Y,(25,1))

lr = LogisticRegression(tol = 10**-8)
lr.fit(X,Y)

test_A = [0,1,2,3,4]
test_B = [0,0,1,1,1]

test_A = numpy.reshape(test_A,(5,1))
predict = lr.predict(test_A)

#added code
mis = accuracy_score(test_A, predict)
"""
###################

data = pandas.read_csv('ChicagoCompletedPotHole.csv')

#loge(N_POTHOLES_FILLED_ON_BLOCK), loge(1 + N_DAYS_FOR_COMPLETION), LATITUDE, and LONGITUDE


trainData = data.to_numpy()
nCity = data.shape[0]

# Determine the number of clusters
maxNClusters = 10

nClusters = numpy.zeros(maxNClusters)
Elbow = numpy.zeros(maxNClusters)
TotalWCSS = numpy.zeros(maxNClusters)
Inertia = numpy.zeros(maxNClusters)

for c in range(maxNClusters):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=2022484).fit(data)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nCity):
      k = kmeans.labels_[i]
      nC[k] += 1
      
      #change diff to be Manhattan Distance instead of Euclidien
      diff = (abs(trainData[i][0] - kmeans.cluster_centers_[k][0])) + (abs(trainData[i][1] - kmeans.cluster_centers_[k][1]))
      WCSS[k] += diff*(diff)

   Elbow[c] = 0
   
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]
      
lis = []
for i in range(Elbow.size):
    lis.append(i+1)

    #plot Elbow Value
plt.clf()
plt.plot(lis, Elbow)
plt.scatter(lis, Elbow)
plt.title('Elbow Value of "TwoFeatures.csv"')
plt.xlabel("Elbow Value")
plt.ylabel("Cluster Number")
plt.show()
      

 
print(metrics.calinski_harabasz_score(nClusters[4], k))
     