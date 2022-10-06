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

count=0
for i in claim["CAR_USE"]:
    if i == "Commercial":
        count+=1
prob_com = count/len(claim["CAR_USE"])

prob_pri = (len(claim["CAR_USE"])-count)/len(claim["CAR_USE"])

entropy = -((prob_com * numpy.log2(prob_com) + prob_pri * numpy.log2(prob_pri)))

#Question 2-----------------------------------------
######################
##Part 1

v10 = pandas.read_csv('sample_v10.csv')
