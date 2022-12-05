#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:52:15 2022

@author: michalmalek
"""

import matplotlib.pyplot as plt
import numpy
import scipy
from scipy import stats as stats
import pandas
import sys
import time
import openpyxl
import sklearn
from sklearn import preprocessing, naive_bayes, metrics, neural_network 
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
import Utility

#Question 1-----------------------------------------
######################

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

####Provided by Professor
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


def cramer(tab):
    chi2 = stats.chi2_contingency(tab)[0]
    n = numpy.sum(tab)
    phi2 = chi2/n
    min_dim = min(tab.shape)-1
    return numpy.sqrt(phi2 / (min_dim))



# Specify the roles
feature = ['group_size', 'homeowner', 'married_couple']
target = 'insurance'


# Get our data
q1 = pandas.read_csv('Purchase_Likelihood.csv',
                              usecols = feature + [target])

q1 = q1.dropna()

# Look at the row distribution
print(q1.groupby(target).size())

for pred in feature:
    RowWithColumn(rowVar = q1[target], columnVar = q1[pred], show = 'ROW')


data1 = numpy.array([[115460,25728,2282,	221], [329552,	91065,	5069,	381], [74293,	19600,	1505,	93]])
data2 = numpy.array([[78659,	65032], [183130,	242937], [46734,	48757]])
data3 = numpy.array([[117110,	26581], [333272,	92795], [75310,	20181]])

Cramer1 = cramer(data1)
Cramer2 = cramer(data2)
Cramer3 = cramer(data3)

subData = q1[['group_size', 'homeowner', 'married_couple', 'insurance']].dropna()

catEBilling = subData['insurance'].unique()
catCreditCard = subData['group_size'].unique()
catGender = subData['homeowner'].unique()
catJobCategory = subData['married_couple'].unique()

print('Unique Values of EBilling: \n', catEBilling)
print('Unique Values of CreditCard: \n', catCreditCard)
print('Unique Values of Gender: \n', catGender)
print('Unique Values of JobCategory: \n', catJobCategory)

RowWithColumn(rowVar = subData['insurance'], columnVar = subData['group_size'], show = 'ROW')
RowWithColumn(rowVar = subData['insurance'], columnVar = subData['homeowner'], show = 'ROW')
RowWithColumn(rowVar = subData['insurance'], columnVar = subData['married_couple'], show = 'ROW')

subData = subData.astype('category')
#xTrain = pandas.get_dummies(subData[['group_size', 'homeowner', 'married_couple']])

yT = numpy.where(subData['insurance'] == 0, 1, 2)

# Correctly Use sklearn.naive_bayes.CategoricalNB
feature = ['group_size', 'homeowner', 'married_couple']

labelEnc = preprocessing.LabelEncoder()
yT = labelEnc.fit_transform(subData['insurance'])

uCreditCard = numpy.unique(subData['group_size'])
uGender = numpy.unique(subData['homeowner'])
uJobCategory = numpy.unique(subData['married_couple'])

featureCategory = [uCreditCard, uGender, uJobCategory]
print(featureCategory)

featureEnc = preprocessing.OrdinalEncoder(categories = featureCategory)
xT = featureEnc.fit_transform(subData[['group_size', 'homeowner', 'married_couple']])

_objNB = naive_bayes.CategoricalNB(alpha = 1.0e-10)
thisModel = _objNB.fit(xT, yT)

xTest = pandas.DataFrame(list((x,y,z) for x in range(4) for y in range(2) for z in range(2)), columns = feature)

# Score the xTest and append the predicted probabilities to the xTest
yTest_predProb = pandas.DataFrame(_objNB.predict_proba(xTest),
                                  columns = ['Insurance 1', 'Insurance 2', 'Insurance 3'])

yTest_score = pandas.concat([xTest, yTest_predProb], axis = 1)

yTest_score.round(6).to_csv('q1.csv')

yTest_score['result'] = yTest_score['Insurance 1']/yTest_score['Insurance 2']

"""

#Question 2-----------------------------------------
######################

print(sklearn.__version__)

q2 = pandas.read_excel('Homeowner_Claim_History.xlsx')

q2['Freq'] = (q2['num_claims']/q2['exposure']) 

yName = 'Freq'

catName = ['f_primary_age_tier','f_primary_gender','f_marital', 'f_residence_location', 'f_fire_alarm_type', 'f_mile_fire_station', 'f_aoi_tier']

trainData = q2[catName + [yName]].dropna().reset_index(drop = True)


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
actFunc = ['identity','tanh']
nLayer = range(1,11,1)
nHiddenNeuron = range(1,6,1)
combList = itertools.product(actFunc, nLayer, nHiddenNeuron)

result_list = []


for comb in combList:
   time_begin = time.time()
   actFunc = comb[0]
   nLayer = comb[1]
   nHiddenNeuron = comb[2]

   nnObj = neural_network.MLPRegressor(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
              activation = actFunc, verbose = False, learning_rate_init = 0.001, max_iter = 10000, random_state = 31010)
   thisFit = nnObj.fit(X_reduce, y)
   
   #Number of Iterations
   n_iter = nnObj.n_iter_
   #y_predProb = pandas.DataFrame(nnObj.predict_proba(X_reduce), columns = y_category)
   
   y_predProb = nnObj.predict(X_reduce)
   
   # Calculate Root Mean Squared Error
   rase = sqrt(mean_squared_error(y, y_predProb))
   
   #Elapsed Time
   elapsed_time = time.time() - time_begin
   
   #Best Loss
   best_loss = nnObj.loss_
   
   #Pearson Correlation
   Pearson_Corr = stats.pearsonr(y, y_predProb)
   
   arr1 = []
   for i in range(len(y_predProb)):
       arr1.append(0)
   
   rel_err = (sklearn.metrics.mean_absolute_error(y, y_predProb))/(sklearn.metrics.mean_absolute_error(y, arr1))
   
   result_list.append([actFunc, nLayer, nHiddenNeuron, n_iter, best_loss, rase, rel_err, Pearson_Corr[0], elapsed_time])
   
   print('Done combo: ', actFunc, nLayer, nHiddenNeuron)

result_df = pandas.DataFrame(result_list, columns = ['Activation Function', 'nLayer', 'nHiddenNeuron', 'nIteration', 'Best Loss', 'RMSE', 'Relative Error', 'Pearson Corr', 'Elapsed Time'])

result_df.round(6).to_csv('q2.csv')

mini = result_df['RMSE'].min()

"""
trueVresult = pandas.read_csv('Q2E.csv')

fig, ax = plt.subplots(dpi = 200, figsize = (10,6))
ax.scatter(trueVresult['Freq'],trueVresult['Pred'])
ax.grid(True)
plt.xlabel("Observed Frequency")
plt.ylabel("Expected Frequency")
plt.title('Observed Frequency VS Predicted Frequency')
plt.show()

"""
plt.figure(figsize=(10,6), dpi = 200)
plt.plot(trueVresult['y'], trueVresult['pearson', 'result'] , linewidth = 2, marker = '')
plt.scatter(trueVresult['y'],trueVresult['pearson'],color='red')
plt.scatter(trueVresult['y'],trueVresult['result'],color='blue')
plt.grid(True)
plt.xlabel("Observed Frequency")
plt.ylabel("Pearson Residual and Simple Residual")
plt.title('Observed Frequency VS Pearson Residual and Simple Residual')
plt.show()
"""

fig, ax = plt.subplots(dpi = 200, figsize = (10,6))
ax.scatter(trueVresult['Freq'],trueVresult['Pearson'],color='red', label= 'Pearson Residual')
ax.scatter(trueVresult['Freq'],trueVresult['Result'],color='blue', label= 'Simple Residual')
ax.grid(True)
plt.legend(loc='upper left')
plt.xlabel("Observed Frequency")
plt.ylabel("Pearson Residual and Simple Residual")
plt.title('Observed Frequency VS Pearson Residual and Simple Residual')
plt.show()



