# -*- coding: utf-8 -*-
"""
@Name: Week 10 Cars Neural Network.py
@Creation Date: March 1, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys
import time

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

import itertools
from sklearn import metrics, neural_network

sys.path.append('C:\\IIT\\Machine Learning\\Job')
import Utility

inputData = pandas.read_csv('cars.csv')

catName = ['Cylinders','Origin','Type']
intName = ['EngineSize','Horsepower','Length','MPG_City','MPG_Highway','Weight','Wheelbase']

yName = 'DriveTrain'

trainData = inputData[catName + intName + [yName]].dropna().reset_index(drop = True)
n_sample = trainData.shape[0]

# Reorder the categories of the target variables in descending frequency
u = trainData[yName].astype('category').copy()
u_freq = u.value_counts(ascending = False)
trainData[yName] = u.cat.reorder_categories(list(u_freq.index)).copy()

# Reorder the categories of the categorical variables in ascending frequency
for pred in catName:
    u = trainData[pred].astype('category').copy()
    u_freq = u.value_counts(ascending = True)
    trainData[pred] = u.cat.reorder_categories(list(u_freq.index)).copy()

X = pandas.get_dummies(trainData[catName].astype('category'))
X = X.join(trainData[intName])
X.insert(0, '_BIAS_', 1.0)

# Identify the aliased parameters
n_param = X.shape[1]
XtX = X.transpose().dot(X)
origDiag = numpy.diag(XtX)
XtXGinv, aliasParam, nonAliasParam = Utility.SWEEPOperator (n_param, XtX, origDiag, sweepCol = range(n_param), tol = 1.0e-7)
X_reduce = X.iloc[:, list(nonAliasParam)].drop(columns = ['_BIAS_'])

y = trainData[yName].astype('category')
y_category = y.cat.categories
n_category = len(y_category)

# Grid Search for the best neural network architecture
actFunc = ['identity','logistic','relu','tanh']
nLayer = range(1,11,1)
nHiddenNeuron = range(5,30,5)
combList = itertools.product(actFunc, nLayer, nHiddenNeuron)

result_list = []

for comb in combList:
   time_begin = time.time()
   actFunc = comb[0]
   nLayer = comb[1]
   nHiddenNeuron = comb[2]

   nnObj = neural_network.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
              activation = actFunc, verbose = False, max_iter = 10000, random_state = 20220301)
   thisFit = nnObj.fit(X_reduce, y)
   n_iter = nnObj.n_iter_
   y_predProb = pandas.DataFrame(nnObj.predict_proba(X_reduce), columns = y_category)

   # Calculate Root Average Squared Error
   rase = 0.0
   for cvalue in y_category:
      y_residual = numpy.where(y == cvalue, 1.0, 0.0) - y_predProb[cvalue]
      rase = rase + numpy.sum(numpy.power(y_residual, 2.0))

   rase = numpy.sqrt(rase / n_category / n_sample)
   elapsed_time = time.time() - time_begin
   result_list.append([actFunc, nLayer, nHiddenNeuron, n_iter, rase, elapsed_time])
   
   print('Done combo: ', actFunc, nLayer, nHiddenNeuron)

result_df = pandas.DataFrame(result_list, columns = ['Activation Function', 'nLayer', 'nHiddenNeuron', 'nIteration', 'RASE', 'Elapsed Time'])

# Review the RASE by each algorithm parameter
fig, ax = plt.subplots(dpi = 200, figsize = (10,4))
result_df.boxplot(column = 'RASE', by = 'Activation Function', ax = ax)
ax.set_xlabel('Activation Function')
ax.set_ylabel('Root Average Squared Error')
plt.suptitle('')
plt.title('')
plt.show()

fig, ax = plt.subplots(dpi = 200, figsize = (10,4))
result_df.boxplot(column = 'RASE', by = 'nLayer', ax = ax)
ax.set_xlabel('Number of Layers')
ax.set_ylabel('Root Average Squared Error')
plt.suptitle('')
plt.title('')
plt.show()

fig, ax = plt.subplots(dpi = 200, figsize = (10,4))
result_df.boxplot(column = 'RASE', by = 'nHiddenNeuron', ax = ax)
ax.set_xlabel('Number of Hidden Neurons per Layer')
ax.set_ylabel('Root Average Squared Error')
plt.suptitle('')
plt.title('')
plt.show()

# Review the Elapsed Time by each algorithm parameteraaaa
fig, ax = plt.subplots(dpi = 200, figsize = (10,4))
ax.hist(result_df['Elapsed Time'], bins = numpy.arange(0.0, 1.0, 0.05), density = True, color = 'dodgerblue')
ax.set_xlabel('Elapsed Time (second)')
ax.set_ylabel('Number of Runs')
ax.set_xticks(numpy.arange(0.0, 1.1, 0.1))
ax.grid(axis = 'y')
plt.show()

print(result_df['Elapsed Time'].describe())

# Locate the optimal architecture
optima_index = result_df['RASE'].idxmin()
optima_row = result_df.iloc[optima_index]
actFunc = optima_row['Activation Function']
nLayer = optima_row['nLayer']
nHiddenNeuron = optima_row['nHiddenNeuron']

# Train the network with the optimal architecture
nnObj = neural_network.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
           activation = actFunc, verbose = False, max_iter = 10000, random_state = 20220301)
thisFit = nnObj.fit(X_reduce, y)
y_predProb = pandas.DataFrame(nnObj.predict_proba(X_reduce), columns = y_category)

y_predClass = y_predProb.idxmax(axis = 'columns')

print(metrics.confusion_matrix(y, y_predClass))

# Review the distributions of the predicted probabilities
cmap = ['red', 'green', 'blue']
fig, axs = plt.subplots(nrows = n_category, ncols = n_category, sharex = True, sharey = False,
                        dpi = 200, figsize = (18,12))
for i in range(n_category):
   obs_value = y_category[i]
   plotData = y_predProb[y == obs_value]
   for j in range(n_category):
      pred_value = y_category[j]
      ax = axs[i,j]
      ax.hist(plotData[pred_value], bins = 10, density = True, facecolor = cmap[i], alpha = 0.75)
      ax.yaxis.grid(True, which = 'major')
      ax.xaxis.grid(True, which = 'major')
      if (i == 0):
         ax.set_title('Pred.Prob.:' + pred_value)
      if (j == 0):
         ax.set_ylabel('Obs.:' + obs_value + '%')
plt.show()
