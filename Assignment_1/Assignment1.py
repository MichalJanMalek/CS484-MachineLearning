#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:59:16 2022

@author: michalmalek
"""


import matplotlib.pyplot as plt
import numpy
import pandas

##Part 1a
df = pandas.read_csv('NormalSample.csv')
X = df['x']

print(df.to_string()) 

print (X.describe())

##Part 1b

print(2*(324-304) * 1000 ** (-1/3))

##Part 1c

plt.hist(X)
plt.show()

def calcCD (X, delta):
   maxX = numpy.max(X)
   minX = numpy.min(X)
   meanX = numpy.mean(X)

   # Round the mean to integral multiples of delta
   middleX = delta * numpy.round(meanX / delta)

   # Determine the number of bins on both sides of the rounded mean
   nBinRight = numpy.ceil((maxX - middleX) / delta)
   nBinLeft = numpy.ceil((middleX - minX) / delta)
   lowX = middleX - nBinLeft * delta

   # Assign observations to bins starting from 0
   m = nBinLeft + nBinRight
   BIN_INDEX = 0;
   boundaryY = lowX
   for iBin in numpy.arange(m):
      boundaryY = boundaryY + delta
      BIN_INDEX = numpy.where(X > boundaryY, iBin+1, BIN_INDEX)

   # Count the number of observations in each bins
   uBin, binFreq = numpy.unique(BIN_INDEX, return_counts = True)

   # Calculate the average frequency
   meanBinFreq = numpy.sum(binFreq) / m
   ssDevBinFreq = numpy.sum((binFreq - meanBinFreq)**2) / m
   CDelta = (2.0 * meanBinFreq - ssDevBinFreq) / (delta * delta)
   return(m, middleX, lowX, CDelta)

result = []
deltaList = [1, 2, 2.5, 5, 10, 20, 25, 50]

for d in deltaList:
   nBin, middleX, lowX, CDelta = calcCD(X,d)
   highX = lowX + nBin * d
   result.append([d, CDelta, lowX, middleX, highX, nBin])

   binMid = lowX + 0.5 * d + numpy.arange(nBin) * d
   plt.hist(X, bins = binMid, align='mid')
   plt.title('Delta = ' + str(d))
   plt.ylabel('Number of Observations')
   plt.grid(axis = 'y')
   plt.show()

result = pandas.DataFrame(result, columns = {0:'Delta', 1:'C(Delta)', 2:'Low Y', 3:'Middle Y', 4:'High Y', 5:'N Bin'})

fig1, ax1 = plt.subplots()
ax1.set_title('Box Plot')
ax1.boxplot(X, labels = ['Y'])
ax1.grid(linestyle = '--', linewidth = 1)
plt.show()
