"""
@Name: Week 6 Cars Logistic All Possible.py
@Creation Date: October 3, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

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

from itertools import combinations

sys.path.append('C:\\IIT\\Machine Learning\\Job')
import Utility

catName = ['DriveTrain']
intName = ['EngineSize', 'Horsepower', 'Length', 'Weight']
yName = 'Origin'

nPredictor = len(catName) + len(intName)

cars = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\cars.csv')

trainData = cars[[yName] + catName + intName].dropna()

allCombResult = []
allFeature = catName + intName

allComb = []
for r in range(nPredictor+1):
   allComb = allComb + list(combinations(allFeature, r))

startTime = time.time()
maxIter = 20
tolS = 1e-7

nComb = len(allComb)
for r in range(nComb):
   modelTerm = list(allComb[r])
   trainData = cars[[yName] + modelTerm].dropna()
   n_sample = trainData.shape[0]

   X_train = trainData[[yName]].copy()
   X_train.insert(0, 'Intercept', 1.0)
   X_train.drop(columns = [yName], inplace = True)

   y_train = trainData[yName].copy()

   for pred in modelTerm:
      if (pred in catName):
         X_train = X_train.join(pandas.get_dummies(trainData[pred].astype('category')))
      elif (pred in intName):
         X_train = X_train.join(trainData[pred])

   resultList = Utility.MNLogisticModel (X_train, y_train, maxIter = maxIter, tolSweep = tolS)

   modelLLK = resultList[1]
   modelDF = resultList[2]
   del resultList

   AIC = 2.0 * modelDF - 2.0 * modelLLK
   BIC = modelDF * numpy.log(n_sample) - 2.0 * modelLLK
   allCombResult.append([r, modelTerm, len(modelTerm), modelLLK, modelDF, AIC, BIC, n_sample])

endTime = time.time()

allCombResult = pandas.DataFrame(allCombResult, columns = ['Step', 'Model Term', 'Number of Terms', \
                'Log-Likelihood', 'Model Degree of Freedom', 'Akaike Information Criterion', \
                'Bayesian Information Criterion', 'Sample Size'])

elapsedTime = endTime - startTime
