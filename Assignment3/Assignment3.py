#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:28:40 2022

@author: michalmalek
"""

import matplotlib.pyplot as plt
import numpy
import math
import pandas
import sklearn
from itertools import combinations
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
import sys
import warnings
warnings.filterwarnings("ignore")


def EntropyIntervalSplit (
   inData,          # input data frame (predictor in column 0 and target in column 1)
   split):          # split value

   #print(split)
   dataTable = inData
   dataTable['LE_Split'] = False
   for k in dataTable.index:
       if dataTable.iloc[:,0][k] in split:
           dataTable['LE_Split'][k] = True
   #print(dataTable['LE_Split'])
   crossTable = pandas.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   
   #print(crossTable)

   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   
   tableEntropy = 0
   for iRow in range(nRows-1):
      rowEntropy = 0
      for iColumn in range(nColumns):
         proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
         if (proportion > 0):
            rowEntropy -= proportion * numpy.log2(proportion)
      #print('Row = ', iRow, 'Entropy =', rowEntropy)
      #print(' ')
      tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
   tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]
  
   return(tableEntropy)



def calculate_min_entropy(df,variable,combinations):
    inData1 = df[[variable,"Labels"]]
    entropies = []
    for i in combinations:
        EV = EntropyIntervalSplit(inData1, list(i))
        entropies.append((EV,i))
    return min(entropies)

#Question 1-----------------------------------------
######################
##Part 1

claim = pandas.read_csv('claim_history.csv')

find = claim[["CAR_TYPE","OCCUPATION","EDUCATION"]]
find.head()

features_train,features_test,labels_train, labels_test = train_test_split(find,claim["CAR_USE"],test_size = 0.3, random_state=27513,stratify = claim["CAR_USE"])

cross_Table_Train = pandas.crosstab(labels_train,columns =  ["Count"],margins=True,dropna=True)
cross_Table_Train["Proportions"] = (cross_Table_Train["Count"]/len(labels_train))*100

cross_Table_test = pandas.crosstab(labels_test,columns =  ["Count"],margins=True,dropna=True)
cross_Table_test["Proportions"] = (cross_Table_test["Count"]/len(labels_test))*100

features_train["Labels"] = labels_train

count=0
for i in claim["CAR_USE"]:
    if i == "Commercial":
        count+=1
prob_com = count/len(claim["CAR_USE"])
prob_pri = (len(claim["CAR_USE"])-count)/len(claim["CAR_USE"])
entropy = -((prob_com * numpy.log2(prob_com) + prob_pri * numpy.log2(prob_pri)))


occupation_column = claim["OCCUPATION"].unique()
occupation_combinations = []
for i in range(1, math.ceil(len(occupation_column)/2)):
    occupation_combinations+=list(combinations(occupation_column,i))
   
    
car_type_column = claim["CAR_TYPE"].unique()
car_type_combinations = []

for i in range(1,math.ceil(len(car_type_column)/2)+1):
    x = list(combinations(car_type_column,i))
    if i == 3:
        x = x[:10]
    car_type_combinations.extend(x) 

education_combinations = [("Below High School",),("Below High School","High School",),("Below High School","High School","Bachelors",),("Below High School","High School","Bachelors","Masters",)]


entropy_occupation = calculate_min_entropy(features_train,"OCCUPATION", occupation_combinations)

entropy_cartype = calculate_min_entropy(features_train,"CAR_TYPE", car_type_combinations)

entropy_education = calculate_min_entropy(features_train,"EDUCATION", education_combinations)


df_1_left = features_train[(features_train["OCCUPATION"] == "Blue Collar") | (features_train["OCCUPATION"] == "Unknown") | (features_train["OCCUPATION"] == "Student")]
df_1_right =  features_train[(features_train["OCCUPATION"] != "Blue Collar") & (features_train["OCCUPATION"] != "Unknown") & (features_train["OCCUPATION"] != "Student")]
len(df_1_right),len(df_1_left)

left_edu_entropy = calculate_min_entropy(df_1_left,"EDUCATION",education_combinations)
left_ct_entropy = calculate_min_entropy(df_1_left,"CAR_TYPE",car_type_combinations)

occupation_column = ['Blue Collar', 'Unknown', 'Student']
occupation_combinations = []
for i in range(1,math.ceil(len(occupation_column)/2)):
    occupation_combinations+=list(combinations(occupation_column,i))
left_occupation_entropy = calculate_min_entropy(df_1_left,"OCCUPATION",occupation_combinations)
occupation_combinations

occupation_column = ['Professional', 'Manager', 'Clerical', 'Doctor','Lawyer','Home Maker']
occupation_combinations = []
for i in range(1,math.ceil(len(occupation_column)/2)):
    occupation_combinations+=list(combinations(occupation_column,i))
right_occupation_entropy = calculate_min_entropy(df_1_right,"OCCUPATION",occupation_combinations)

right_edu_entropy = calculate_min_entropy(df_1_right,"EDUCATION",education_combinations)
right_ct_entropy = calculate_min_entropy(df_1_right,"CAR_TYPE",car_type_combinations)

df_2_left_left = df_1_left[(features_train["EDUCATION"] == "Below High School")]
df_2_left_right = df_1_left[(features_train["EDUCATION"] != "Below High School")]

cnt = 0
for i in df_2_left_left["Labels"]:
    if i == "Commercial":
        cnt+=1
proba_commercial1 = cnt/len(df_2_left_left["Labels"])

cnt = 0
for i in df_2_left_right["Labels"]:
    if i == "Commercial":
        cnt+=1
proba_commercial2 = cnt/len(df_2_left_right["Labels"])

df_2_right_left = df_1_right[(features_train["CAR_TYPE"] == "Minivan") | (features_train["CAR_TYPE"] == "Sports Car") | (features_train["CAR_TYPE"] == "SUV")]
df_2_right_right = df_1_right[(features_train["CAR_TYPE"] != "Minivan") & (features_train["CAR_TYPE"] != "Sports Car") & (features_train["CAR_TYPE"] != "SUV")]

cnt = 0
for i in df_2_right_left["Labels"]:
    if i == "Commercial":
        cnt+=1
proba_commercial3 = cnt/len(df_2_right_left["Labels"])
1-proba_commercial3

cnt = 0
for i in df_2_right_right["Labels"]:
    if i == "Commercial":
        cnt+=1
proba_commercial4 = cnt/len(df_2_right_right["Labels"])


                          


#Question 2-----------------------------------------
######################
##Part a-c

import Utility

intName = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
yName = 'y'

nPredictor = len(intName)

cars = pandas.read_csv('sample_v10.csv')

trainData = cars[[yName] + intName].dropna()
del cars

n_sample = trainData.shape[0]

# Frequency of the nominal target
print('=== Frequency of ' + yName + ' ===')
print(trainData[yName].value_counts())

# Specify the color sequence
cmap = ['indianred','sandybrown','royalblue']

# Reorder the categories of the target variables in descending frequency
u = trainData[yName].astype('category').copy()
u_freq = u.value_counts(ascending = False)
trainData[yName] = u.cat.reorder_categories(list(u_freq.index)).copy()

# Generate a column of Intercept
X0_train = trainData[[yName]].copy()
X0_train.insert(0, 'Intercept', 1.0)
X0_train.drop(columns = [yName], inplace = True)

y_train = trainData[yName].copy()

maxIter = 20
tolS = 1e-7
stepSummary = []

# Intercept only model
resultList = Utility.MNLogisticModel (X0_train, y_train, maxIter = maxIter, tolSweep = tolS)

llk0 = resultList[1]
df0 = resultList[2]
stepSummary.append(['Intercept', ' ', df0, llk0, numpy.NaN, numpy.NaN, numpy.NaN])

#cName = catName.copy()
iName = intName.copy()
entryThreshold = 0.05

# The Deviance significance is the sixth element in each row of the test result
def takeDevSig(s):
    return s[6]

for step in range(nPredictor):
    enterName = ''
    stepDetail = []


    for X_name in iName:
        X_train = trainData[[X_name]]
        X_train = X0_train.join(X_train)
        resultList = Utility.MNLogisticModel (X_train, y_train, maxIter = maxIter, tolSweep = tolS)
        llk1 = resultList[1]
        df1 = resultList[2]
        devChiSq = 2.0 * (llk1 - llk0)
        devDF = df1 - df0
        devSig = chi2.sf(devChiSq, devDF)
        stepDetail.append([X_name, 'interval', df1, llk1, devChiSq, devDF, devSig])

    # Find a predictor to enter, if any
    # Find a predictor to add, if any
    stepDetail.sort(key = takeDevSig, reverse = False)
    enterRow = stepDetail[0]
    minPValue = takeDevSig(enterRow)
    if (minPValue <= entryThreshold):
        stepSummary.append(enterRow)
        df0 = enterRow[2]
        llk0 = enterRow[3]

        enterName = enterRow[0]
        enterType = enterRow[1]
        if (enterType == 'interval'):
            X_train = trainData[[enterName]]
            X0_train = X0_train.join(X_train)
            iName.remove(enterName)
    else:
        break

    print('======= Step Detail =======')
    print('Step = ', step+1)
    print('Step Statistics:')
    print(stepDetail)
    print('Enter predictor = ', enterName)
    print('Minimum P-Value =', minPValue)
    print('\n')
# End of forward selection

stepSummary = pandas.DataFrame(stepSummary, columns = ['Predictor', 'Type', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig'])

######################
##Part e

modelLLK = stepSummary['ModelLLK']
modelDF = stepSummary['ModelDF']
del resultList

AIC = 2.0 * modelDF - 2.0 * modelLLK
BIC = modelDF * numpy.log(n_sample) - 2.0 * modelLLK

