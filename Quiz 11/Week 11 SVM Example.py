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

trainData = pandas.read_excel('C:\\IIT\\Machine Learning\\Data\\MVN.xlsx', sheet_name = 'MVN')

# Scatterplot that uses prior information of the grouping variable
carray = ['red', 'green']
plt.figure(figsize = (8,6), dpi=200)
for i in range(2):
    subData = trainData[trainData['Group'] == i]
    plt.scatter(x = subData['X'], y = subData['Y'], c = carray[i], label = i, s = 25)
plt.xticks(numpy.arange(-3.0, 4.0, 1.0))
plt.yticks(numpy.arange(-3.0, 4.0, 1.0))
plt.grid(True)
plt.title('Prior Group Information')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Build Support Vector Machine classifier
trainData = trainData.dropna().reset_index(drop = True)
xTrain = trainData[['X','Y']]
yTrain = trainData['Group']

svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = 20221225)
thisFit = svm_Model.fit(xTrain, yTrain)
yTrain_pred = thisFit.predict(xTrain)

print('Number of Iteration =', thisFit.n_iter_)
print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

w = thisFit.coef_[0]

print('=== General Equation Form ===')
eq_str = str(thisFit.intercept_[0]) + ' + ' + str(w[0]) + ' * X + ' + str(w[1]) + ' * Y'
print(eq_str)

# Convert to y = a + mx form
coef_m = - numpy.divide(w[0], w[1])
coef_a = numpy.divide(thisFit.intercept_[0], w[1]) 

print('=== Slope-Intercept Equation Form ===')
eq_str = 'Y = ' + str(coef_a) + ' + ' + str(coef_m) + ' * X'
print(eq_str)

print('\n=== Confusion Matrix === ')
print(metrics.confusion_matrix(yTrain, yTrain_pred))

# plot the parallels to the separating hyperplane that pass through the
# support vectors

eps_double = numpy.finfo(numpy.float64).eps
trainData['Decision'] = thisFit.decision_function(xTrain)
trainData['ABS Decision'] = trainData['Decision'].apply(numpy.abs)
support_vectors = trainData[trainData['ABS Decision'] <= (eps_double + 1.0)]
support_vectors = support_vectors.sort_values(by = ['Group', 'ABS Decision'])

outmost_sv = support_vectors.groupby('Group').last()

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'green']
plt.figure(figsize = (10,8), dpi = 200)
for i in range(2):
    plot_data = trainData[yTrain_pred == i]
    plt.scatter(x = plot_data['X'], y = plot_data['Y'], c = carray[i], label = 'Predicted Group' + str(i), s = 25)
plt.scatter(support_vectors['X'], support_vectors['Y'], s=100, linewidth=1, facecolors="none",  edgecolors="k")

# Draw the boundaries
x_value = outmost_sv.iloc[0]['X']
y_value = outmost_sv.iloc[0]['Y']
plt.axline((x_value, y_value), slope = coef_m, linestyle = 'dashed', color = 'red', label = 'Group 0 Boundary')    

x_value = outmost_sv.iloc[-1]['X']
y_value = outmost_sv.iloc[-1]['Y']
plt.axline((x_value, y_value), slope = coef_m, linestyle = 'dashed', color = 'green', label = 'Group 1 Boundary')    

# Draw the separating hyperplane
x_value = numpy.mean(support_vectors['X'])
y_value = coef_a + coef_m * x_value
plt.axline((x_value, y_value), slope = coef_m, linestyle = 'solid', color = 'black', label = 'Separator')    
    
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

