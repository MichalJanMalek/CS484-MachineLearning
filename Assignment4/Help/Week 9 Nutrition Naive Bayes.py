"""
@Name: Week 9 Week 9 Nutrition Naive Bayes.py
@Creation Date: August 22, 2022
@Author: Ming-Long Lam, Ph.D.
@Organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

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

import itertools
from sklearn import preprocessing, naive_bayes

# Define a function to visualize the percent of a particular target category by a nominal predictor
def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

   countTable = pandas.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
   print("Frequency Table: \n", countTable)
   print( )

   if (show == 'ROW' or show == 'BOTH'):
       rowFraction = countTable.div(countTable.sum(1), axis='index')
       print("Row Fraction Table: \n", rowFraction)
       print( )

   if (show == 'COLUMN' or show == 'BOTH'):
       columnFraction = countTable.div(countTable.sum(0), axis='columns')
       print("Column Fraction Table: \n", columnFraction)
       print( )

   return

# Specify the roles
feature = ['tv', 'magazine', 'friends', 'doctor']
target = 'supps'

# Read the Excel file
nutrition = pandas.read_excel('Nutrition_Information.xls',
                              sheet_name = 'Sheet1',
                              usecols = feature + [target])

nutrition = nutrition.dropna()

# Look at the row distribution
print(nutrition.groupby(target).size())

for pred in feature:
    RowWithColumn(rowVar = nutrition[target], columnVar = nutrition[pred], show = 'ROW')

# Make the binary features take values 0 and 1 (was 2=No and 1=Yes)
nutrition[feature] = 2 - nutrition[feature]

xTrain = nutrition[feature].astype('category')
yTrain = nutrition[target].astype('category')

_objNB = naive_bayes.BernoulliNB(alpha = 1.e-10)
thisFit = _objNB.fit(xTrain, yTrain)

print('Probability of each class')
print(numpy.exp(thisFit.class_log_prior_))

print('Empirical probability of features given a class, P(x_i|y)')
print(numpy.exp(thisFit.feature_log_prob_))

print('Number of samples encountered for each class during fitting')
print(thisFit.class_count_)

print('Number of samples encountered for each (class, feature) during fitting')
print(thisFit.feature_count_)

yTrain_predProb = _objNB.predict_proba(xTrain)

# Create the all possible combinations of the features' values
xTest = pandas.DataFrame(list(itertools.product([0,1], repeat = len(feature))), columns = feature)

# Score the xTest and append the predicted probabilities to the xTest
yTest_predProb = pandas.DataFrame(_objNB.predict_proba(xTest), columns = ['P_suppsYes', 'P_suppsNo'])
yTest_score = pandas.concat([xTest, yTest_predProb], axis = 1)
