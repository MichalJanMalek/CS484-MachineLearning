# -*- coding: utf-8 -*-
"""
@ame: Week 12 Toy AdaBoost Example.py
@creation Date: November 21, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

from sklearn import (metrics, tree)

X_train = pandas.DataFrame({'x1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9],
                            'x2': [0.3, 0.2, 0.1, 0.4, 0.7, 0.5, 0.9, 0.8, 0.2, 0.8]})

y_train = pandas.Series([1, 1, 0, 0, 1, 0, 1, 1, 0, 0], name = 'y')

carray = ['blue', 'red']
plt.figure(figsize=(16,9), dpi = 200)
for j in range(2):
   subset_data = X_train[y_train == j]
   plt.scatter(subset_data['x1'], subset_data['x2'], c = carray[j], label = str(j), s = 100)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.legend(title = 'y', fontsize = 12, markerscale = 1)
plt.show()

# Suppose no limit on the maximum number of depths
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=60616)
treeFit = classTree.fit(X_train, y_train)
y_predProb = classTree.predict_proba(X_train)
y_predClass = numpy.where(y_predProb[:,1] >= 0.5, 1, 0)

confusion_matrix = metrics.confusion_matrix(y_train, y_predClass)
print(confusion_matrix)

fig, ax = plt.subplots(1, 1, figsize = (8,12), dpi = 200)
tree.plot_tree(classTree, max_depth=10, feature_names=['x1', 'x2'], class_names=['0', '1'], label='all', filled=True, impurity=True, ax=ax)
plt.show()

carray = ['blue', 'red']
plt.figure(figsize=(16,9), dpi = 200)
for j in range(2):
   subset_data = X_train[y_train == j]
   plt.scatter(subset_data['x1'], subset_data['x2'], c = carray[j], label = str(j), s = 100)
plt.plot([0.25, 0.25], [0, 1], color = 'red', linestyle = ':')
plt.plot([0.85, 0.85], [0, 1], color = 'red', linestyle = ':')
plt.plot([0, 1], [0.6, 0.6], color = 'red', linestyle = ':')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.legend(title = 'y', fontsize = 12, markerscale = 1)
plt.show()

# Build a classification tree on the training partition
w_train = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype = float)

y_ens_predProb = numpy.zeros((len(y_train), 2))
ens_accuracy = 0.0

for itnum in range(6):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=60616)
    treeFit = classTree.fit(X_train, y_train, w_train)
    y_predProb = classTree.predict_proba(X_train)
    y_predClass = numpy.where(y_predProb[:,1] >= 0.5, 1, 0)
    accuracy = numpy.sum(numpy.where(y_train == y_predClass, w_train, 0.0)) / numpy.sum(w_train)
    ens_accuracy = ens_accuracy + accuracy
    y_ens_predProb = y_ens_predProb + accuracy * y_predProb

    print('\n')
    print('Iteration = ', itnum)
    print('Weighted Accuracy = ', accuracy)
    print('Weight:\n', w_train)
    print('Predicted Class:\n', y_predClass)

    if (abs(1.0 - accuracy) < 0.0000001):
        break

    # Update the weights
    eventError = numpy.where(y_train == 1, (1 - y_predProb[:,1]), (0 - y_predProb[:,1]))
    w_train = numpy.abs(eventError)
    w_train = numpy.where(y_predClass != y_train, 1.0, 0.0) + w_train

    print('Event Error:\n', eventError)

y_ens_predProb = y_ens_predProb / ens_accuracy