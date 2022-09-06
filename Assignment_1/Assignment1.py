#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:59:16 2022

@author: michalmalek
"""


import matplotlib.pyplot as plt
import numpy
import pandas

#Question 1-----------------------------------------
######################
##Part 1
df = pandas.read_csv('NormalSample.csv')
X = df['x']

#print(df.to_string()) 

print (X.describe())

######################
##Part 2

print ("\n" + "Bin Width : "+"%.1f" % round((2*(324-304) * 1000 ** (-1/3)), 1)+ '\n')

######################
##Part 3

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


#Question 2-----------------------------------------

dataSet0 = df[df['group']==0]
print (dataSet0.describe().loc[['min', '25%', '50%', '75%', 'max']])


dataSet1 = df[df['group']==1]
print (dataSet1.describe().loc[['min', '25%', '50%', '75%', 'max']])

#Question 3-----------------------------------------

q3 = pandas.read_csv('Fraud.csv')

#print(q3.to_string()) 

fraudCases = "{:.4f}".format(((((q3['FRAUD'] == 1).sum())/5960)*100), 4) + "%"

             
print("\n" + fraudCases + '\n')

#Question 4-----------------------------------------
######################
##Part 1
q4 = pandas.read_csv('flights.csv').fillna('')
print(q4.to_string()) 

Xax = q4['Airport 2']
Yax = q4['Airport 3']

plt.scatter(Xax, Yax)
plt.ylabel('Airport 3')
plt.xlabel('Airport 2')

plt.show()

#######################
##Part 2

s1 = pandas.Series(q4['Airport 2'].values)
s2 = pandas.Series(q4['Airport 3'].values)
combined = pandas.concat([s1, s2])
freq = combined.replace({''}, numpy.nan).value_counts()

print (freq)

#######################
##Part 3

#scipy.spatial.distance.cosine(row,word_count_list[i])

