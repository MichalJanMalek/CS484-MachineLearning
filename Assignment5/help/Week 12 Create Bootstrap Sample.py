# -*- coding: utf-8 -*-
"""
@ame: Week 12 Create Bootstrap Sample.py
@creation Date: November 14, 2022
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

import random

# Create a bootstrap sample from the population
def sample_wr (inData):
    n = len(inData)
    outData = numpy.empty((n,1))
    for i in range(n):
        j = int(random.random() * n)
        outData[i] = inData[j]
    return outData

x = numpy.array([1,2,3,4,5,6,7,8,10])
unique, counts = numpy.unique(x, return_counts = True)
print('Original Sample:\n', numpy.asarray((unique, counts)).transpose())

# Check the bootstrap sample
random.seed(20191113)

sample1 = sample_wr(x)
unique, counts = numpy.unique(sample1, return_counts = True)
print('Sample 1:\n', numpy.asarray((unique, counts)).transpose())

sample2 = sample_wr(x)
unique, counts = numpy.unique(sample2, return_counts = True)
print('Sample 2:\n', numpy.asarray((unique, counts)).transpose())

sample3 = sample_wr(x)
unique, counts = numpy.unique(sample3, return_counts = True)
print('Sample 3:\n', numpy.asarray((unique, counts)).transpose())

sample4 = sample_wr(x)
unique, counts = numpy.unique(sample4, return_counts = True)
print('Sample 4:\n', numpy.asarray((unique, counts)).transpose())
