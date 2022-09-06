#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:59:16 2022

@author: michalmalek
"""


import matplotlib.pyplot as plt
import numpy 
from numpy import linalg as LA
import scipy
from scipy import linalg as LA2
import pandas
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

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

######################
##Part 4
d=10

nBin, middleX, lowX, CDelta = calcCD(X,d)
highX = lowX + nBin * d
result.append([d, CDelta, lowX, middleX, highX, nBin])

binMid = lowX + 0.5 * d + numpy.arange(nBin) * d
plot = plt.hist(X, bins = binMid, align='mid')
plt.title('Delta = ' + str(d))
plt.ylabel('Number of Observations')
plt.grid(axis = 'y')
plt.show()

mid = plot[1]
print(mid)
newArray = []
for i in mid:
    newArray.append(i+5)
    
print (newArray)    

d = plot[0]
newerArray = []
for i in d:
    newerArray.append(i/(10000))

newerArray.append(0)

plt.bar(x=newArray, height=newerArray, width=5)    
plt.title('Estimated Density Function')
plt.ylabel('p(m)')
plt.grid(axis='y')
plt.show()
    

#Question 2-----------------------------------------
######################
##Part 1

    #split up dataset into two groups
dataSet0 = df[df['group']==0]
dataSet1 = df[df['group']==1]

print(dataSet0)

    #format for 5-number summary
set0 = dataSet0.describe().loc[['min', '25%', '50%', '75%', 'max']]['x']
set1 = dataSet1.describe().loc[['min', '25%', '50%', '75%', 'max']]['x']

    #get IQR and whiskers for set 0
q1s0 = dataSet0.describe().loc['25%'].tolist()[1]
q3s0 = dataSet0.describe().loc['75%'].tolist()[1]
iqr0 = (q3s0 - q1s0)
lower0 = q1s0 - 1.5*iqr0
upper0 = q3s0 + 1.5*iqr0

    #get IQR and whiskers for set 1
q1s1 = dataSet1.describe().loc['25%'].tolist()[1]
q3s1 = dataSet1.describe().loc['75%'].tolist()[1]
iqr1 = (q3s1 - q1s1)
lower1 = q1s1 - 1.5*iqr1
upper1 = q3s1 + 1.5*iqr1

    #print 5-number sum out + whisker
print ('')
print (set0)
print ("\nLower Whisker = ",lower0,"\nUpper Whisker = ",upper0)
print ('')
print (set1)
print ("\nLower Whisker = ",lower1,"\nUpper Whisker = ",upper1)

######################
##Part 2

fig1, ax1 = plt.subplots()
ax1.set_title('Overall Box Plot')
ax1.boxplot([X, dataSet0['x'], dataSet1['x']], labels = ['Overall', 'Group 0', 'Group 1'])
ax1.grid(linestyle = '--', linewidth = 1)
plt.show()

#find outliers by finding all that are lower than Q1 nd greater than Q3

set1 = dataSet1.describe().loc[['min', '25%', '50%', '75%', 'max']]['x']

    #get IQR and whiskers for set 0
q1A = X.describe().loc['25%'].tolist()

q3A = X.describe().loc['75%'].tolist()
iqA = (q3A - q1A)
print (iqA)
lowerAll = q1A - 1.5*iqA
print (lowerAll)
upperAll = q3A + 1.5*iqA
print (upperAll)

allOut = []
g1Out = []
g2Out = []

for i in X:
    if i < lowerAll or i > upperAll:
        allOut.append(i)

for j in dataSet0['x']:
    if j < lower0 or j > upper0:
        g1Out.append(j)
    
for k in dataSet1['x']:
    if k < lower1 or k > upper1:
        g2Out.append(k)
    
print ('Outlier for Both : ' + str(allOut))
print ('Outlier for Group 0 : ' + str(g1Out))
print ('Outlier for Group 1 : ' + str(g2Out))     
       
       

#Question 3-----------------------------------------
######################
##Part 1

q3 = pandas.read_csv('Fraud.csv')

#print(q3.to_string()) 

fraudCases = ("{:.4f}".format(((((q3['FRAUD'] == 1).sum())/5960)*100), 4) + "%")

             
print("\nPercent of Frauduelent Cases : " + fraudCases + '\n')

######################
##Part 2

x = q3.copy().drop(columns=['CASE_ID', 'FRAUD']).to_numpy()
    #code provided by professor
#
xtx = numpy.matmul(x.transpose(), x)
print("t(x) * x = \n", xtx)

    # Eigenvalue decomposition
evals, evecs = LA.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)

    # Want eigenvalues greater than one
evals_1 = evals[evals > 1.0]
evecs_1 = evecs[:,evals > 1.0]

    # Here is the transformation matrix
dvals = 1.0 / numpy.sqrt(evals_1)
transf = numpy.matmul(evecs_1, numpy.diagflat(dvals))
print("Transformation Matrix = \n", transf)

    # Here is the transformed X
transf_x = numpy.matmul(x, transf)
print("The Transformed x = \n", transf_x)

    # Check columns of transformed X
xtx = numpy.matmul(transf_x.transpose(), transf_x)
print("Expect an Identity Matrix = \n", xtx)

######################
##Part 3
y = q3['FRAUD']
target_class = list(y.unique())
k=5

neigh = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')
nbrs = neigh.fit(transf_x, y)
class_prob = nbrs.predict_proba(transf_x)

nbrs_list = numpy.argmax(class_prob, axis = 1)
predicted_class = [target_class[k] for k in nbrs_list]
rate_miss_class = numpy.mean(numpy.where(predicted_class == y, 0, 1))

miss = ("{:.4f}".format((rate_miss_class*100), 4) + "%")
print(miss)

nbrs = ("{:.4f}".format((nbrs.score(transf_x, y)*100), 4) + "%")
print(nbrs)
######################
##Part 4

print(q3.median())
median = [[16300, 8, 0, 178, 1 ,2]]

med_transf = numpy.matmul(median, transf)

neighborsfive = neigh.kneighbors(med_transf, return_distance=False)
print(neighborsfive)

print (q3.iloc[neighborsfive[0]])



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
freq = combined.replace({''}, '___').value_counts()

#print (freq)

#######################
##Part 3

ap = freq.index
count = []
for i in q4.drop(columns=['Flight', 'Carrier 1', 'Carrier 2', 'Airport 1', 'Airport 4']).iterrows():
    inrow = []
    for j in ap:
        inrow.append(list(i[1]).count(j))
    count.append(inrow)
    
print(count)

newFly = ['LAX', '___']
newRow = []
for i in ap:
    newRow.append((newFly).count(i))

cosList = []
for i in range(13):
    cosList.append(scipy.spatial.distance.cosine(newRow, count[i]))

print(cosList)
