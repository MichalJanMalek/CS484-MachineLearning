#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:59:16 2022

@author: michalmalek
"""


import pandas


df = pandas.read_csv('NormalSample.csv')
Xline = df['x']

print(df.to_string()) 

print (Xline.describe())

