#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:28:40 2022

@author: michalmalek
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn

from sklearn.model_selection import train_test_split

#Question 1-----------------------------------------
######################
##Part 1

claim = pandas.read_csv('claim_history.csv')

find = claim[["CAR_TYPE","OCCUPATION","EDUCATION"]]

find_train, find_test, labels_train, labels_test = train_test_split(find,claim["CAR_USE"],test_size = 0.3, random_state=27513,stratify = claim["CAR_USE"])

cross_Table_Train = pandas.crosstab(labels_train,columns =  ["Count"],margins=True,dropna=True)
cross_Table_Train["Proportions"] = (cross_Table_Train["Count"]/len(labels_train))*100

cross_Table_Test = pandas.crosstab(labels_test ,columns =  ["Count"],margins=True,dropna=True)
cross_Table_Test["Proportions"] = (cross_Table_Train["Count"]/len(labels_test))*100

#Question 2-----------------------------------------
######################
##Part 1

v10 = pandas.read_csv('sample_v10.csv')
