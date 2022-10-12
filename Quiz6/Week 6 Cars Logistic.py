"""
@Name: Week 6 Cars Logistic.py
@Creation Date: October 3, 2022
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

import sklearn.metrics as metrics
import statsmodels.api as smodel

sys.path.append('C:\\IIT\\Machine Learning\\Job')
import Utility

cars = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\cars.csv')
n_sample = cars.shape[0]

# Specify Origin as a categorical variable
y = cars['Origin'].astype('category')
y_category = y.cat.categories
print('Categories of Origin:', y_category)
print(y.value_counts())

# Train a model with only the Intercept term
X0 = cars[['Origin']].copy()
X0.insert(0, 'Intercept', 1.0)
X0.drop(columns = ['Origin'], inplace = True)

# Train a multinominal logistic model
modelObj = smodel.MNLogit(y, X0)

print("Name of Target Variable:", modelObj.endog_names)
print("Name(s) of Predictors:", modelObj.exog_names)

thisFit = modelObj.fit(full_output = True)
print('Model Summary:\n', thisFit.summary())

print('Intercept Model Log-Likelihood Value = ', thisFit.llnull)
print('  Current Model Log-Likelihood Value = ', thisFit.llf)

print("Model Parameter Estimates:\n", thisFit.params)

y_predProb = thisFit.predict(X0)
y_predProb.columns = y_category
y_predClass = y_predProb.idxmax(axis = 1)

y_confusion = pandas.DataFrame(metrics.confusion_matrix(y, y_predClass))
y_confusion.columns = y_category
y_confusion.index = y_category
print("Confusion Matrix (Row is Observed, Column is Predicted) = \n")
print(y_confusion)

y_accuracy = metrics.accuracy_score(y, y_predClass)
print("Accuracy Score = ", y_accuracy)

# Review the predicted probabilities
plotData = y_predProb.join(y)
fig, axs = plt.subplots(nrows = 1, ncols = 3, sharex = True, sharey = True, dpi = 200, figsize=(12,6))
for j in range(3):
    ax = axs[j]
    plotData.boxplot(column = y_category[j], by = 'Origin', ax = ax, vert = True)
    # ax.set_ylabel('Predicted Probability for ' + str(y_category[j]))
    ax.set_xlabel('Origin')
    ax.yaxis.grid(True)
    ax.set_title(y_category[j])
plt.suptitle('Predicted Probability for\n')
plt.show()

# Add the continuous predictors EngineSize, Horsepower, Length, and Weight
X = X0.join(cars[['EngineSize','Horsepower','Length','Weight']])

# Train a multinominal logistic model
modelObj = smodel.MNLogit(y, X)

print("Name of Target Variable:", modelObj.endog_names)
print("Name(s) of Predictors:", modelObj.exog_names)

thisFit = modelObj.fit(full_output = True)
print('Model Summary:\n', thisFit.summary())

print('Intercept Model Log-Likelihood Value = ', thisFit.llnull)
print('  Current Model Log-Likelihood Value = ', thisFit.llf)

print("Model Parameter Estimates:\n", thisFit.params)

y_predProb = thisFit.predict(X)
y_predProb.columns = y_category
y_predClass = y_predProb.idxmax(axis = 1)

y_confusion = pandas.DataFrame(metrics.confusion_matrix(y, y_predClass))
y_confusion.columns = y_category
y_confusion.index = y_category
print("Confusion Matrix (Row is Observed, Column is Predicted) = \n")
print(y_confusion)

y_accuracy = metrics.accuracy_score(y, y_predClass)
print("Accuracy Score = ", y_accuracy)

# Review the predicted probabilities
plotData = y_predProb.join(y)
fig, axs = plt.subplots(nrows = 1, ncols = 3, sharex = True, sharey = True, dpi = 200, figsize=(12,6))
for j in range(3):
    ax = axs[j]
    plotData.boxplot(column = y_category[j], by = 'Origin', ax = ax, vert = True)
    # ax.set_ylabel('Predicted Probability for ' + str(y_category[j]))
    ax.set_xlabel('Origin')
    ax.yaxis.grid(True)
    ax.set_title(y_category[j])
plt.suptitle('Predicted Probability for\n')
plt.show()

# Add the categorical predictors DriveTrain
print(cars[['DriveTrain']].astype('category').value_counts())
X = X0.join(pandas.get_dummies(cars[['DriveTrain']].astype('category')))

# Identify the aliased parameters
XtX = X.transpose().dot(X)
n_param = X.shape[1]
origDiag = numpy.diag(XtX)
XtXGinv, aliasParam, nonAliasParam = Utility.SWEEPOperator (n_param, XtX, origDiag)
print('Aliased Parameters: ', list(aliasParam))
print('Non-aliased Parameters: ', list(nonAliasParam))

# Train a multinominal logistic model
X_reduce = X.iloc[:, list(nonAliasParam)]
modelObj = smodel.MNLogit(y, X_reduce)

print("Name of Target Variable:", modelObj.endog_names)
print("Name(s) of Predictors:", modelObj.exog_names)

thisFit = modelObj.fit(full_output = True)
print('Model Summary:\n', thisFit.summary())

print('Intercept Model Log-Likelihood Value = ', thisFit.llnull)
print('  Current Model Log-Likelihood Value = ', thisFit.llf)

print("Model Parameter Estimates:\n", thisFit.params)

y_predProb = thisFit.predict(X_reduce)
y_predProb.columns = y_category
y_predClass = y_predProb.idxmax(axis = 1)

y_confusion = pandas.DataFrame(metrics.confusion_matrix(y, y_predClass))
y_confusion.columns = y_category
y_confusion.index = y_category
print("Confusion Matrix (Row is Observed, Column is Predicted) = \n")
print(y_confusion)

y_accuracy = metrics.accuracy_score(y, y_predClass)
print("Accuracy Score = ", y_accuracy)

# Review the predicted probabilities
plotData = y_predProb.join(y)
fig, axs = plt.subplots(nrows = 1, ncols = 3, sharex = True, sharey = True, dpi = 200, figsize=(12,6))
for j in range(3):
    ax = axs[j]
    plotData.boxplot(column = y_category[j], by = 'Origin', ax = ax, vert = True)
    # ax.set_ylabel('Predicted Probability for ' + str(y_category[j]))
    ax.set_xlabel('Origin')
    ax.yaxis.grid(True)
    ax.set_title(y_category[j])
plt.suptitle('Predicted Probability for\n')
plt.show()
