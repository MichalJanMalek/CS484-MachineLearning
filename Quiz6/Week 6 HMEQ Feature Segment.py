"""
@Name: Week 6 HMEQ Feature Segment.py
@Creation Date: October 3, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import pandas

# Set some options for printing all the columns
pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

# Define a function to visualize the percent of a particular target category by a nominal predictor
def TargetPercentByNominal (
   targetVar,       # target variable
   predictor):      # nominal predictor

   countTable = pandas.crosstab(index = predictor, columns = targetVar, margins = True, dropna = True)
   x = countTable.drop(labels = 'All', axis = 1)
   percentTable = countTable.div(x.sum(1), axis='index')*100
   oddsTable = percentTable.div(percentTable[0], axis='index')
   overallOdds = oddsTable.loc['All'][1].item()
   oddsTable['Ratio'] = oddsTable[1] / overallOdds
   oddsTable.drop(columns = [0, 'All'], inplace = True)

   print("\nFrequency Table:")
   print(countTable)
   print("\nPercent Table:")
   print(percentTable)
   print("\nOdds Table:")
   print(oddsTable)

   return (countTable, percentTable, oddsTable)
   
hmeq = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\hmeq.csv')

countTable, percentTable, oddsTable = TargetPercentByNominal(hmeq['BAD'], hmeq['REASON'])

countTable, percentTable, oddsTable = TargetPercentByNominal(hmeq['BAD'], hmeq['JOB'])

countTable, percentTable, oddsTable = TargetPercentByNominal(hmeq['BAD'], [hmeq['REASON'], hmeq['JOB']])

countTable, percentTable, oddsTable = TargetPercentByNominal(hmeq['BAD'], [hmeq['REASON'], hmeq['JOB'], hmeq['DEROG']])

