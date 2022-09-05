#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 23:29:02 2022

@author: michalmalek
"""
import numpy
from sklearn.neighbors import KNeighborsClassifier
import pandas

cars = pandas.read_csv('cars.csv',delimiter=',')

cars["CaseID"] = cars["Make"] + "_" + cars.index.values.astype(str)

cars_wIndex = cars.set_index("CaseID")

##cars_wIndex = pandas.read_csv('cars.csv')


trainData = cars_wIndex[['Invoice', 'Horsepower', 'Weight', 'Origin']].dropna().reset_index()
X = trainData[['Invoice', 'Horsepower', 'Weight']]
y = trainData['Origin']
target_class = list(y.unique())
neigh_choice = range(5,20,1)
result = []

for k in neigh_choice:
    neigh = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')
    nbrs = neigh.fit(X, y)
    class_prob = nbrs.predict_proba(X)

    nbrs_list = numpy.argmax(class_prob, axis = 1)
    predicted_class = [target_class[k] for k in nbrs_list]
    rate_miss_class = numpy.mean(numpy.where(predicted_class == y, 0, 1))
    result.append([k, rate_miss_class])

miss_classification_k = pandas.DataFrame(result, columns = ['Number of Neighbors', 'Misclassification Rate'])
