#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:10:51 2022

@author: michalmalek
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys
import seaborn as sns


# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format
 
from sklearn import (ensemble, metrics, model_selection, tree)

trainData = pandas.read_csv('WineQuality_Train.csv')

n_sample = trainData.shape[0]



y_threshold = numpy.mean(trainData['quality_grp'])

X_train = trainData[['alcohol','citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_train = trainData['quality_grp']

# Suppose no limit on the maximum number of depths
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101 )
treeFit = classTree.fit(X_train, y_train)
y_predProb = classTree.predict_proba(X_train)
y_predClass = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)
confusion_matrix = metrics.confusion_matrix(y_train, y_predClass)
print(confusion_matrix)

# Build a classification tree on the training partition
max_iteration = 50
w_train = numpy.full(n_sample, 1.0)

ens_accuracy = numpy.zeros(max_iteration)
y_ens_predProb = numpy.zeros((n_sample, 2))

for itnum in range(max_iteration):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101 )
    treeFit = classTree.fit(X_train, y_train, w_train)
    y_predProb = classTree.predict_proba(X_train)
    y_predClass = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)
    accuracy = numpy.sum(numpy.where(y_train == y_predClass, w_train, 0.0)) / numpy.sum(w_train)
    ens_accuracy[itnum] = accuracy
    y_ens_predProb = y_ens_predProb + accuracy * y_predProb

    print('\n')
    print('Iteration = ', itnum)
    print('\nWeighted Accuracy = ', accuracy)
    #print('Weight:\n', w_train)
    #print('Predicted Class:\n', y_predClass)
    print('\nMisclassification Rate = ', 1- accuracy)

    if (accuracy >= 0.9999999):
        break

    # Update the weights
    eventError = numpy.where(y_train == 1, (1 - y_predProb[:,1]), (y_predProb[:,1]))
    w_train = numpy.abs(eventError)
    MAE = numpy.abs(eventError)
    w_train = numpy.where(y_predClass != y_train, MAE+2, MAE)

    #print('Event Error:\n', eventError)

y_ens_predProb = y_ens_predProb / numpy.sum(ens_accuracy)

# Calculate the final predicted probabilities
trainData['predCluster'] = numpy.where(y_ens_predProb[:,1] >= y_threshold, 1, 0)
ensembleAccuracy = numpy.mean(numpy.where(trainData['predCluster'] == y_train, 1, 0))

classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101)
boostTree = ensemble.AdaBoostClassifier(base_estimator=classTree, n_estimators=28,
                                        learning_rate=1.0, algorithm='SAMME.R', random_state=20230101)
boostFit = boostTree.fit(X_train, y_train)
boostPredProb = boostFit.predict_proba(X_train)
boostAccuracy = boostFit.score(X_train, y_train)

trainData['predCluster'] = numpy.where(boostPredProb[:,1] >= y_threshold, 1, 0)

 # The column that holds the dependent variable's values
AUC = metrics.roc_auc_score(y_train, y_predProb[:,1])

print('Area Under Curve = ', AUC)

sns.boxplot(data=trainData, x = trainData['quality_grp'], y= y_predProb[:,1], width = .3)






####################
#Q2

def calc_auc(f, X, y):
    y_pp = f.predict(X)
    y_p = pandas.to_numeric(y_pp.idxmax(axis=1))
    return metrics.roc_auc_score(y, y_p)

import random
import time
import sklearn
from sklearn import metrics, svm, utils
import pandas
import sklearn.ensemble as ensemble
import statsmodels.api as stats

WineQuality = pandas.read_csv('WineQuality_Train.csv')

WineQuality = WineQuality.dropna()

WQ_size = WineQuality.groupby('quality_grp').size()

X_name = ['alcohol', 'free_sulfur_dioxide', 'sulphates', 'citric_acid', 'residual_sugar']

# Build a logistic regression
y = WineQuality['quality_grp'].astype('category')
y_category = y.cat.categories

X = WineQuality[X_name]
X = stats.add_constant(X, prepend=True)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(maxiter = 100)
thisParameter = thisFit.params

y_predProb = thisFit.predict(X)
y_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))

y_predictClass = y_category[y_predict]

AUC = metrics.roc_auc_score(y, y_predictClass)

print('\nArea Under Curve = ', AUC)

nB = 100000
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=20221225)
bagTree = ensemble.BaggingClassifier(base_estimator = classTree, n_estimators = nB, random_state = 20221225, verbose = 1)
bagFit = bagTree.fit(X, y)
bagPredProb = bagFit.predict_proba(X)


AUCs = []
n_iterations = 100000
for i in range(n_iterations):
    print(i)
    X_bs, y_bs = sklearn.utils.resample(X, y, replace=True)
    logit = stats.MNLogit(y_bs, X_bs)
    thisFit = logit.fit(disp=False)
    AUCs.append(calc_auc(thisFit, X, y))

plt.clf()
sns.histplot(AUCs, binwidth=0.001)
plt.title("AUC across 100,000 bootstrap samples")
plt.xlabel("AUC")
plt.show()


percentile_values = [2.5, 50, 97.5]
percentiles = numpy.percentile(AUCs, percentile_values)
for i in range(len(percentile_values)): print(str(percentile_values[i]) + 'th Percentile:', percentiles[i])
"""





objSVM = svm.LinearSVC(verbose = 1, dual = False, max_iter = 1000, random_state = 20221225)
thisFit = objSVM.fit(X_train, y_train)
print('Intercept = ', thisFit.intercept_)

y_train_pred = thisFit.predict(X_train)

AUC = metrics.roc_auc_score(y_train, y_train_pred)

print('Area Under Curve = ', AUC)

df = pandas.read_csv('WineQuality_Train.csv')

X_train = df[['alcohol', 'free_sulfur_dioxide', 'sulphates', 'citric_acid', 'residual_sugar']]
y_train = df['quality_grp']

objSVM = svm.LinearSVC(verbose = 1, dual = False, max_iter = 1000, random_state = 20221225)
thisFit = objSVM.fit(X_train, y_train)
#print('Intercept = ', thisFit.intercept_)

y_test_pred = thisFit.predict(X_train)

AUC = metrics.roc_auc_score(y_train, y_test_pred)
print('\nArea Under Curve = ', AUC)



threshold = len(y_train[y_train == 1]) / len(y_train)

nB = 100000
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=20221225)
bagTree = ensemble.BaggingClassifier(base_estimator = classTree, n_estimators = nB, random_state = 20191113, verbose = 1)
bagFit = bagTree.fit(X_train, y_train)
bagPredProb = bagFit.predict_proba(X_train)

AUC_test, RASE_test, MisClassRate_test = ModelMetrics (y_train, 1, bagPredProb[:,1], threshold)

print('      Number of Bootstraps: ', nB)
print('          Area Under Curve: {:.7f}' .format(AUC_test))
print('    Misclassification Rate: {:.7f}' .format(MisClassRate_test))
print('Root Average Squared Error: {:.7f}' .format(RASE_test))

#Find AUC metric of the Training data
y_predProb = thisFit.predict(X_train)
y_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))
import sklearn.metrics
AUC = sklearn.metrics.roc_auc_score(y_train, y_predict)
print('Train AUC =', AUC)



WineQuality = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\WineQuality.csv', delimiter=',')

WineQuality = WineQuality.dropna()

WQ_size = WineQuality.groupby('quality_grp').size()

X_name = ['fixed_acidity', 'citric_acid', 'residual_sugar', 'free_sulfur_dioxide',
          'total_sulfur_dioxide', 'pH', 'sulphates']

# Build a logistic regression
y = WineQuality['quality_grp'].astype('category')
y_category = y.cat.categories

X = WineQuality[X_name]
X = stats.add_constant(X, prepend=True)

logit = stats.MNLogit(y, X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(maxiter = 100)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProb = thisFit.predict(X)
y_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))

y_predictClass = y_category[y_predict]

y_confusion = metrics.confusion_matrix(y, y_predictClass)
print("Confusion Matrix (Row is Data, Column is Predicted) = \n")
print(y_confusion)

y_accuracy = metrics.accuracy_score(y, y_predictClass)
print("Accuracy Score = ", y_accuracy)

# Perform the Gradient Boosting
X = WineQuality[X_name]
gbm = ensemble.GradientBoostingClassifier(loss='deviance', criterion='squared_error', n_estimators = 1000,
                                          max_leaf_nodes = 10, verbose=1)
fit_gbm = gbm.fit(X, y)
predY_gbm = gbm.predict(X)
Accuracy_gbm = gbm.score(X, y)
print('Gradient Boosting Model:', 'Accuracy = ', Accuracy_gbm)

# Plot the data
y_confusion = metrics.confusion_matrix(y, predY_gbm)
print("Confusion Matrix (Row is True, Column is Predicted) = \n")
print(y_confusion)

"""







