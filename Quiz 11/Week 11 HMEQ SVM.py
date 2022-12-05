# -*- coding: utf-8 -*-
"""
@ame: Week 11 SVM Example.py
@creation Date: November 14, 2022
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

from sklearn import metrics, svm

hmeq = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\HMEQ.csv', delimiter=',')

y_name = 'BAD'

# Set dual = False because n_samples > n_features

# Step 1
accuracyResult = []
includeVar = []
X_name = ['CLAGE','CLNO','DELINQ','DEROG','NINQ','YOJ']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult.append([includeVar + [ivar], inputData.shape[0], y_predictAccuracy])

print('Step 1')
print(pandas.DataFrame(accuracyResult))

# Step 2
accuracyResult = []
includeVar = ['YOJ']
X_name = ['CLAGE','CLNO','DELINQ','DEROG','NINQ']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult.append([includeVar + [ivar], inputData.shape[0], y_predictAccuracy])

print('Step 2')
print(pandas.DataFrame(accuracyResult))

# Step 3
accuracyResult = []
includeVar = ['YOJ', 'NINQ']
X_name = ['CLAGE','CLNO','DELINQ','DEROG']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult.append([includeVar + [ivar], inputData.shape[0], y_predictAccuracy])

print('Step 3')
print(pandas.DataFrame(accuracyResult))

# Step 4
accuracyResult = []
includeVar = ['YOJ', 'NINQ', 'CLNO']
X_name = ['CLAGE','DELINQ','DEROG']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult.append([includeVar + [ivar], inputData.shape[0], y_predictAccuracy])

print('Step 4')
print(pandas.DataFrame(accuracyResult))

# Step 5
accuracyResult = []
includeVar = ['YOJ', 'NINQ', 'CLNO', 'CLAGE']
X_name = ['DELINQ','DEROG']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult.append([includeVar + [ivar], inputData.shape[0], y_predictAccuracy])

print('Step 5')
print(pandas.DataFrame(accuracyResult))

# Step 6
accuracyResult = []
includeVar = ['YOJ', 'NINQ', 'CLNO', 'CLAGE', 'DEROG']
X_name = ['DELINQ']

for ivar in X_name:
    inputData = hmeq[includeVar + [y_name, ivar]].dropna()
    X = inputData[includeVar + [ivar]]
    y = inputData[y_name].astype('category')
    svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = None, max_iter = 10000)
    thisFit = svm_Model.fit(X, y)
    y_predictClass = thisFit.predict(X)
    y_predictAccuracy = metrics.accuracy_score(y, y_predictClass)
    accuracyResult.append([includeVar + [ivar], inputData.shape[0], y_predictAccuracy])

print('Step 6')
print(pandas.DataFrame(accuracyResult))
