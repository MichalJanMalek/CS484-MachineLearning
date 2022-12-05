# -*- coding: utf-8 -*-
"""
@Name: Week 10 Toy Neural Network.py
@Creation Date: March 1, 2022
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

import sklearn
print(sklearn.__version__)

from sklearn import metrics, neural_network


def pieceWise (x):
    if (x < 0.49):
        y = numpy.sqrt(x)
    else:
        y = 0.7 + 12.68742791 * (x - 0.49)**2
    return y

x = numpy.arange(0.0, 1.005, 0.005)
y = numpy.array([pieceWise(xi) for xi in x])

# Plot the toy data
plt.figure(figsize=(10,6), dpi = 200)
plt.plot(x, y, linewidth = 2, marker = '')
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

xVar = pandas.DataFrame(x, columns = ['x'])

result_list = []

for nLayer in numpy.arange(1,10):
    fig, axs = plt.subplots(nrows = 1, ncols = 4, sharex = True, sharey = True, dpi = 200, figsize=(20,3))
    j = 0
    for nHiddenNeuron in numpy.arange(5,25,5):

        # Build Neural Network
        nnObj = neural_network.MLPRegressor(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                   activation = 'relu', verbose = False, learning_rate_init = 0.001,
                   max_iter = 10000, random_state = 20200301)

        thisFit = nnObj.fit(xVar, y)
        y_pred = nnObj.predict(xVar)

        Loss = nnObj.loss_
        N_Iteration = nnObj.n_iter_
        RSquare = metrics.r2_score(y, y_pred)

        result_list.append([nLayer, nHiddenNeuron, N_Iteration, Loss, RSquare])

        # Plot the prediction
        ax = axs[j]
        ax.scatter(xVar, y, marker = 'o', c = 'black', s = 10, label = 'Data')
        ax.scatter(xVar, y_pred, marker = '.', c = 'red', s = 5, label = 'Prediction')
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("%d Hidden Layers, %d Hidden Neurons" % (nLayer, nHiddenNeuron))
        j = j + 1
    plt.legend(fontsize = 12, markerscale = 3)
    plt.show()

result_df = pandas.DataFrame(result_list, columns = ['N Layer', 'N Hidden Neuron', 'N Iteration', 'Loss', 'RSquare'])
