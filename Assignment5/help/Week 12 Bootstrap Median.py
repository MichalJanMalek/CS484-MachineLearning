# -*- coding: utf-8 -*-
"""
@ame: Week 12 Boostrap Median.py
@creation Date: November 21, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import random
import sys
import time

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

from matplotlib.ticker import FormatStrFormatter

from numpy.random import default_rng
from scipy.stats import norm

# Set the random seed
rng = default_rng(20220901)

# Specifications
n_sample = 101
normal_mu = 10
normal_std = 2

# Generate X from a Normal with mean = 10 and sd = 7
x_sample = norm.rvs(loc = normal_mu, scale = normal_std, size = n_sample, random_state = rng)
print('Sample Median: {:.7f}'.format(numpy.median(x_sample)))

for n_trial in [100, 1000, 10000, 100000, 1000000]:
   time_begin = time.time()

   random.seed(a = 20221225)
   boot_result = [numpy.nan] * n_trial

   for i_trial in range(n_trial):
      boot_index = [-1] * n_sample
      for i in range(n_sample):
         j = int(random.random() * n_sample)
         boot_index[i] = j
      boot_sample = x_sample[boot_index]
      boot_result[i_trial] = numpy.median(boot_sample)

   elapsed_time = time.time() - time_begin

   print('\n')
   print('Number of Trials: ', n_trial)
   print('Elapsed Time: {:.7f}'.format(elapsed_time))
   
   print('Bootstrap Statistics:')
   print('                 Number:', n_trial)
   print('Number of Failed Trials:', numpy.sum(numpy.isnan(boot_result)))
   print('                   Mean: {:.7f}' .format(numpy.mean(boot_result)))
   print('     Standard Deviation: {:.7f}' .format(numpy.std(boot_result, ddof = 1)))
   print('95% Confidence Interval: {:.7f}, {:.7f}'
         .format(numpy.percentile(boot_result, (2.5)), numpy.percentile(boot_result, (97.5))))

   fig, ax = plt.subplots(1,1,figsize = (8,6), dpi = 200)
   ax.hist(boot_result, density = True, align = 'mid', bins = 50)
   ax.set_title('Number of Bootstraps = ' + str(n_trial))
   ax.set_xlabel('Medians of Boostrap Samples')
   ax.set_ylabel('Percent of Bootstrap Samples')
   ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
   plt.grid(axis = 'both')
   plt.show()
