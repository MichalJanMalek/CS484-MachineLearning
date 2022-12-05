"""
@Name: Week 9 Cuisine Naive Bayes.py
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

from sklearn import naive_bayes

# Columns are Carolina, French, Korean, New York, Philly, Texas,
#             Barbecue, Macaron, Souffle, Toast, Steak
X = numpy.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], dtype = float)

y = numpy.array([1,0,0,1,0,0,1,1,1], dtype = float)

classifier = naive_bayes.MultinomialNB(alpha = 1).fit(X, y)
print('Alpha Value = ', classifier.alpha)

print('Class Count:\n', classifier.class_count_)
print('Log Class Probability:\n', classifier.class_log_prior_ )
print('Feature Count (before adding alpha):\n', classifier.feature_count_)
print('Log Feature Probability:\n', classifier.feature_log_prob_)

predProb = classifier.predict_proba(X)
print('Predicted Conditional Probability (Training):', predProb)

X_test = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]])

print('Predicted Conditional Probability (Testing):\n', classifier.predict_proba(X_test))
