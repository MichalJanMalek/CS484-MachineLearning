#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:49:43 2022

@author: michalmalek
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys
import seaborn as sns
from sklearn import metrics, svm, preprocessing, naive_bayes, ensemble, model_selection, tree
import statsmodels.api as stats

import warnings
warnings.filterwarnings("ignore")
Answers = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#Q1

q1 = pandas.read_csv('Q1.csv')

X= q1['x']

plot = plt.hist(X, bins = 5, align='mid')
plt.ylabel('Number of Observations')
plt.grid(axis = 'y')
plt.show()


Answers[0] = 0.018
#Q2

print (X.describe())

q1s0 = q1.describe().loc['25%'].tolist()[0]
q3s0 = q1.describe().loc['75%'].tolist()[0]
iqr0 = (q3s0 - q1s0)

print(iqr0)

Answers[1] = iqr0
#Q3

cluster_num = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

elbows = numpy.array([579857.9543, 532455.2722, 493218.0813, 433215.8150, 430290.4574, 412804.9312, 409729.7423, 404285.7518, 378087.1355, 369686.6227])

plt.clf()
plt.plot(cluster_num, elbows)
plt.scatter(cluster_num, elbows)
plt.title('Elbow Value of "TwoFeatures.csv"')
plt.xlabel("Cluster Number")
plt.ylabel("Elbow Value")
plt.show()

Answers[2] = 4
#Q4

Answers[3] = 0.04
#Q5

Answers[4] = 0.5952
#Q6

Answers[5] = 0.875
#Q7

Answers[6] = 'E'
#Q8

Answers[7] = 'D'
#Q9

Answers[8] = 'C'
#Q10

Answers[9] = 'A'
#Q11

Answers[10] = 0.7416
#Q12

q12 = pandas.read_csv('Q12.csv')



Answers[11] = 1.1881
#Q13

Answers[12] = '-2+1u-1v=0'
#Q14

Answers[13] = 3
#Q15

q15 = pandas.read_csv('Q15.csv')

# Convert to the polar coordinates
q15['radius'] = numpy.sqrt(q15['x']**2 + q15['y']**2)

q15['theta'] = numpy.arctan2(q15['y'], q15['x'])

# ArcTan2 gives angle from –Pi to Pi
# Make the angle from 0 to 2*Pi
def customArcTan (z):
    theta = numpy.where(z < 0.0, 2.0*numpy.pi+z, z)
    return (theta)

q15['theta'] = q15['theta'].apply(customArcTan)

# Build Support Vector Machine classifier
xTrain = q15[['x','y']]
yTrain = q15['group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191106, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
q15['_PredictedClass_'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)


Answers[14] = 'x*x+y*y-3.4051=0'
#Q16

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

inumpyutData = pandas.read_csv('Q12.csv')

# EBilling -> CreditCard, Gender, JobCategory
subData = inumpyutData[['CreditCard', 'Gender', 'MaritalStatus', 'Retired']].dropna()

catRetired = subData['Retired'].unique()
catCreditCard = subData['CreditCard'].unique()
catGender = subData['Gender'].unique()
catMaritalStatus = subData['MaritalStatus'].unique()

print('Unique Values of Retired: \n', catRetired)
print('Unique Values of CreditCard: \n', catCreditCard)
print('Unique Values of Gender: \n', catGender)
print('Unique Values of MaritalStatus: \n', catMaritalStatus)

RowWithColumn(rowVar = subData['CreditCard'], columnVar = subData['MaritalStatus'], show = 'ROW')
RowWithColumn(rowVar = subData['CreditCard'], columnVar = subData['Gender'], show = 'ROW')
RowWithColumn(rowVar = subData['CreditCard'], columnVar = subData['Retired'], show = 'ROW')

subData = subData.astype('category')

# Correctly Use sklearn.naive_bayes.CategoricalNB
feature = ['Retired', 'Gender', 'MaritalStatus']

labelEnc = preprocessing.LabelEncoder()
yTrain = labelEnc.fit_transform(subData['CreditCard'])
yLabel = labelEnc.inverse_transform([0, 1])

uRetired = numpy.unique(subData['Retired'])
uGender = numpy.unique(subData['Gender'])
uMaritalStatus = numpy.unique(subData['MaritalStatus'])

featureCategory = [uRetired, uGender, uMaritalStatus]
print(featureCategory)

featureEnc = preprocessing.OrdinalEncoder(categories = featureCategory)
xTrain = featureEnc.fit_transform(subData[['Retired', 'Gender', 'MaritalStatus']])

_objNB = naive_bayes.CategoricalNB(alpha = 1.0e-10)
thisModel = _objNB.fit(xTrain, yTrain)

print('Number of samples encountered for each class during fitting')
print(yLabel)
print(_objNB.class_count_)
print('\n')

print('Probability of each class:')
print(yLabel)
print(numpy.exp(_objNB.class_log_prior_))
print('\n')

print('Number of samples encountered for each (class, feature) during fitting')
for i in range(3):
   print('Feature: ', feature[i])
   print(featureCategory[i])
   print(_objNB.category_count_[i])
   print('\n')

print('Empirical probability of features given a class, P(x_i|y)')
for i in range(3):
   print('Feature: ', feature[i])
   print(featureCategory[i])
   print(numpy.exp(_objNB.feature_log_prob_[i]))
   print('\n')

# CreditCard = American Express, Gender = Female, JobCategory = Professional
xTest = featureEnc.transform([['Yes', 'Female', 'Married']])

y_predProb = thisModel.predict_proba(xTest)
print('Predicted Probability: ', yLabel, y_predProb)


Answers[15] = '[Gender = Male], [MaritalStatus = Married], [Retired = No]'
#Q17

Answers[16] = 'Data but no answer yet/Dont know waht to do with data'
#Q18
lst1 =	[0, 	0, 	0, 	0, 	1, 	1 ,	1 ,	1 	,2 	,2 	,2 ,	2 ,	3 ,	3 ,	3 ,	3 ,	4 ,	4 ,	4 ,	4 ]
ylst1 =	[0, 	0, 	0, 	0,	0 ,	0 ,	0 	,1 ,	0 ,	0 	,1 ,	1 ,	0 ,	1 ,	1 ,	1 ,	1 ,	1 ,	1 ,	1 ]

x1= pandas.DataFrame(lst1)
y1= pandas.DataFrame(ylst1)


#y = WineQuality['quality_grp'].astype('category')

y_category = y1[0]

# Build a logistic regression
logit = stats.MNLogit(y1, x1)
thisFit = logit.fit(maxiter = 100)
thisParameter = thisFit.params

y_predProb = thisFit.predict(x1)
y_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))

y_predictClass = y_category[y_predict]


# Build a classification tree on the training partition
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=60616)



x2 = 	[0, 	1, 	2, 	3, 	4] 
y2 = 	[0, 	0, 	1, 	1, 	1] 


#1.	Classification Tree, the splitting criterion is Entropy, the maximum depth is 5, and the random state is 60616. 
#2.	Logistic, Intercept term included, the optimization method is newton, the maximum number of iterations is 100, the tolerance level is 1e-8. 


Answers[17] = 'Data but no answer yet/Dont know waht to do with data'
#Q19

trainData = pandas.read_csv('Face.csv')
IS_NOSE = 0

# Build Support Vector Machine classifier
trainData = trainData.dropna().reset_index(drop = True)
xTrain = trainData[['x','y']]

yTrain = trainData['feature']

svm_Model = svm.LinearSVC(verbose = 1, dual = False, random_state = 20221225)
thisFit = svm_Model.fit(xTrain, yTrain)
yTrain_pred = thisFit.predict(xTrain)

print('Number of Iteration =', thisFit.n_iter_)
print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

w = thisFit.coef_[0]
#Intercept =  [-1.1982150284 -1.1982150284 -0.5174090484 -2.6927052561]
#Coefficients =  [[-2.8217493258e+00  1.3610475516e+00]
# [ 2.8217493258e+00  1.3610475516e+00]
# [ 2.5367797840e-18 -2.3605326191e-01]
# [-3.7501673818e-17 -1.9705260490e+00]]

Answers[18] = 0.9380
#Q20

trainData = trainData.dropna()

WQ_size = trainData.groupby('feature').size()

X_name = ['x', 'y']

# Build a logistic regression
y = trainData['feature'].astype('category')
y_category = y.cat.categories

X = trainData[X_name]
X = stats.add_constant(X, prepend=True)

logit = stats.MNLogit(y, X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(maxiter = 100)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProb = thisFit.predict(X)

# Perform the Gradient Boosting
X = trainData[X_name]
gbm = ensemble.GradientBoostingRegressor(n_estimators = 1000,max_leaf_nodes = 4, verbose=1, random_state=20221225)
fit_gbm = gbm.fit(X, y)
predY_gbm = gbm.predict(X)
Accuracy_gbm = gbm.score(X, y)
print('Gradient Boosting Model:', 'Accuracy = ', Accuracy_gbm)

Answers[19] = 0.7871
###final answers


count = 0
for x in range(len(Answers)):
    count += 1
    print("\n",count)
    print("Answer = ", Answers[x])










df = pandas.read_csv('claim.csv')
print(df)

from sklearn.linear_model import LogisticRegression

X_name = ['Yes', 'No']

X = df[X_name]

y = df['Child']


lr = stats.MNLogit(y, X)

nex = lr.fit(maxiter = 100)
thisParameter = nex.params

w = nex.coef_
y_pred = nex.predict_proba(X)[:, 1]

X.to_numpy() 

y.to_numpy() 

def full_log_likelihood(w, X, y):
    score = numpy.dot(X, w).reshape(1, X.shape[0])
    return numpy.sum(-numpy.log(1 + numpy.exp(score))) + numpy.sum(y * score)

def null_log_likelihood(w, X, y):
    z = numpy.array([w if i == 0 else 0.0 for i, w in enumerate(w.reshape(1, X.shape[1])[0])]).reshape(X.shape[1], 1)
    score = numpy.dot(X, z).reshape(1, X.shape[0])
    return numpy.sum(-numpy.log(1 + numpy.exp(score))) + numpy.sum(y * score)

def mcfadden_rsquare(w, X, y):
    return 1.0 - (full_log_likelihood(w, X, y) / null_log_likelihood(w, X, y))

def mcfadden_adjusted_rsquare(w, X, y):
    k = float(X.shape[1])
    return 1.0 - ((full_log_likelihood(w, X, y) - k) / null_log_likelihood(w, X, y))

print(mcfadden_rsquare(w, X, y))
