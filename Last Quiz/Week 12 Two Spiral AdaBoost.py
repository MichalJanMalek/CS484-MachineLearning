# -*- coding: utf-8 -*-
"""
@ame: Week 12 Two Spiral AdaBoost.py
@creation Date: November 21, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys
import sklearn.metrics

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

from sklearn import (ensemble, metrics, tree)

trainData = pandas.read_csv('SpiralWithCluster.csv')

n_sample = trainData.shape[0]

carray = ['red', 'blue']
plt.figure(figsize=(10,10), dpi = 200)
for i in range(2):
    subData = trainData[trainData['SpectralCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Spiral with Cluster')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'SpectralCluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 15)
plt.show()

y_threshold = 0.5

X_train = trainData[['x','y']]
y_train = trainData['SpectralCluster']

# Suppose no limit on the maximum number of depths
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=60616)
treeFit = classTree.fit(X_train, y_train)
y_predProb = classTree.predict_proba(X_train)
y_predClass = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)
confusion_matrix = metrics.confusion_matrix(y_train, y_predClass)
print(confusion_matrix)

fig, ax = plt.subplots(1, 1, figsize = (16,16), dpi = 200)
tree.plot_tree(classTree, max_depth=10, feature_names=['x', 'y'], class_names=['0', '1'], label='all', filled=True, impurity=True, ax=ax)
plt.show()

# Build a classification tree on the training partition
max_iteration = 30
w_train = numpy.full(n_sample, 1.0)

ens_accuracy = numpy.zeros(max_iteration)
y_ens_predProb = numpy.zeros((n_sample, 2))

for itnum in range(max_iteration):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)
    treeFit = classTree.fit(X_train, y_train, w_train)
    y_predProb = classTree.predict_proba(X_train)
    y_predClass = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)
    accuracy = numpy.sum(numpy.where(y_train == y_predClass, w_train, 0.0)) / numpy.sum(w_train)
    ens_accuracy[itnum] = accuracy
    y_ens_predProb = y_ens_predProb + accuracy * y_predProb

    print('\n')
    print('Iteration = ', itnum)
    print('Weighted Accuracy = ', accuracy)
    print('Weight:\n', w_train)
    print('Predicted Class:\n', y_predClass)

    MAE = metrics.mean_absolute_error(y_predProb, X_train)
    AE = abs(X_train - y_predProb)

    if (abs(1.0 - accuracy) < 0.5):
        MAE = MAE + AE
        break
    

    # Update the weights
    #eventError = numpy.where(y_train == 1, (1 - y_predProb[:,1]), (0 - y_predProb[:,1]))
    w_train = numpy.abs(MAE)
    w_train = numpy.where(y_predClass != y_train, 1.0, 0.0) + w_train

    print('Event Error:\n', MAE)

y_ens_predProb = y_ens_predProb / numpy.sum(ens_accuracy)

# Calculate the final predicted probabilities
trainData['predCluster'] = numpy.where(y_ens_predProb[:,1] >= y_threshold, 1, 0)
ensembleAccuracy = numpy.mean(numpy.where(trainData['predCluster'] == y_train, 1, 0))

carray = ['red', 'blue']
plt.figure(figsize=(10,10), dpi = 200)
for i in range(2):
    subData = trainData[trainData['predCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Spiral with Cluster')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Cluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 15)
plt.show()

classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)
boostTree = ensemble.AdaBoostClassifier(base_estimator=classTree, n_estimators=28,
                                        learning_rate=1.0, algorithm='SAMME.R', random_state=None)
boostFit = boostTree.fit(X_train, y_train)
boostPredProb = boostFit.predict_proba(X_train)
boostAccuracy = boostFit.score(X_train, y_train)

trainData['predCluster'] = numpy.where(boostPredProb[:,1] >= y_threshold, 1, 0)

carray = ['red', 'blue']
plt.figure(figsize=(10,10), dpi = 200)
for i in range(2):
    subData = trainData[trainData['predCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Spiral with Cluster')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Cluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 15)
plt.show()
