#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:24:03 2022

@author: michalmalek
"""
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.metrics as metrics

from mlxtend.frequent_patterns import (apriori, association_rules)
from mlxtend.preprocessing import TransactionEncoder



####################
## Part 4
# a ---------------------

    #creates silhouette widths at each point
sil = metrics.silhouette_samples([(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (6, 0), (7, 0), (8, 0), (9, 0)],[0, 0, 0, 0, 0, 1, 1, 1, 1])

    #since 1 would be the second value
silWidthOf2 = sil[1] 

# b ---------------------



# c ---------------------

    #this will output the Davies-Bouldin Index of the two clusters
dbIndex = metrics.davies_bouldin_score([(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (6, 0), (7, 0), (8, 0), (9, 0)], [0, 0, 0, 0, 0, 1, 1, 1, 1])


####################
##Part 5
# a ---------------------

shop = pandas.read_csv('Groceries.csv')

ListItem = shop.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)
nCustomer, nProduct = ItemIndicator.shape

itemsets = apriori(ItemIndicator, min_support = 75/9835, use_colnames = True)

item = []
for i in range(0,len(itemsets["itemsets"])):
    item.append(len(itemsets["itemsets"][i]))
    
largestKvalue = max(item)

assoc = association_rules(itemsets, metric = "confidence", min_threshold = 0.01)

plt.figure(figsize=(6,4))

plt.scatter(assoc['confidence'], assoc['support'], s = assoc['lift'], c = assoc['lift'])
plt.colorbar().set_label('Lift')
plt.title("Support VS Confidence")
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

sixtyPercentRules = association_rules(itemsets, metric = "confidence", min_threshold = 0.6)


####################
##Part 6
# a ---------------------

    #get data from TwoFeatures
data = pandas.read_csv('TwoFeatures.csv')


    #get x1 from TwoFeature
x1 = data['x1']

    #get x2 from TwoFeature
x2 = data['x2']

    #create scatterplot x1 vs x2

plt.scatter(x1, x2)
plt.title("Scatterplot of x1 vs. x2")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# b ---------------------

    #this set of code is modified from professor's given "Distance From Chicago.py"

trainData = data.to_numpy()
nCity = data.shape[0]

# Determine the number of clusters
maxNClusters = 8

nClusters = numpy.zeros(maxNClusters)
Elbow = numpy.zeros(maxNClusters)
TotalWCSS = numpy.zeros(maxNClusters)
Inertia = numpy.zeros(maxNClusters)

for c in range(maxNClusters):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=484000).fit(data)

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

# c ---------------------

    #make list the size of Elbow to get correct cluster number because we need to add 1
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

# d ---------------------


newX1 = []
newX2 = []


for x in range(x1.size):
    newX1.append(((10*(x1[x]-min(x1)))/(max(x1)-min(x1))))
    
    
for x in range(x2.size):
    newX2.append((((10)*(x2[x]-min(x2)))/(max(x2)-min(x2))))
    

plt.clf()
plt.scatter(newX1, newX2)
plt.grid()
plt.title("Scatterplot of x1 vs. x2 After Scaling")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

    
d = {'x1': newX1, 'x2': newX2}
df = pandas.DataFrame(data=d)

trainData = df.to_numpy()
nCity = df.shape[0]

# Determine the number of clusters
maxNClusters = 8

nClusters = numpy.zeros(maxNClusters)
Elbow = numpy.zeros(maxNClusters)
TotalWCSS = numpy.zeros(maxNClusters)
Inertia = numpy.zeros(maxNClusters)

for c in range(maxNClusters):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=484000).fit(df)

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
      
# e ---------------------
      
lis = []
for i in range(Elbow.size):
    lis.append(i+1)
    
plt.clf()
plt.plot(lis, Elbow)
plt.scatter(lis, Elbow)
plt.title('Elbow Value of "TwoFeatures.csv" After Scaling')
plt.xlabel("Elbow Value")
plt.ylabel("Cluster Number")
plt.show()
