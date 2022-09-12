"""
@Name: Week 2 Sample Nearest Neighbor.py
@Creation Date: August 29, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas

rank_vector = lambda vec: list(map(lambda i: sorted(vec).index(i)+1, vec))

def dist_2_data (probe, df):
    out_list = []
    for index, row in df.iterrows():
        euclidean = numpy.sqrt(numpy.sum(numpy.power((row - probe), 2)))
        out_list.append(euclidean)
    out_distance = pandas.DataFrame([out_list], columns = df.index)
    out_rank = pandas.DataFrame([rank_vector(out_list)], columns = df.index)
    return(out_distance, out_rank)

def pair_euclidean (df):
    out_distance = pandas.DataFrame()
    out_rank = pandas.DataFrame()
    for index, row in df.iterrows():
        row_distance, row_rank = dist_2_data(row, df)
        out_distance = pandas.concat([out_distance, row_distance], axis = 0, ignore_index = True)
        out_rank = pandas.concat([out_rank, row_rank], axis = 0, ignore_index = True)
    out_distance = out_distance.set_index(df.index)
    out_rank = out_rank.set_index(df.index)
    return(out_distance, out_rank)

def find_neighbor (row_rank, n_neighbor):
    row_columns = row_rank.columns
    row_neighbor = []
    for index, row in row_rank.iterrows():
        row_neighbor.append([row_columns[row == (i+1)][0] for i in range(n_neighbor)])
    out_neighbor = pandas.DataFrame(row_neighbor).set_index(row_rank.index)
    return(out_neighbor)

def find_neighbor_distance (pairwise_distance, row_neighbor):
    row_neighbor_distance = []
    for index, row in pairwise_distance.iterrows():
        row_neighbor_distance.append(list(row[row_neighbor.loc[index]]))
    out_neighbor_distance = pandas.DataFrame(row_neighbor_distance).set_index(pairwise_distance.index)
    return(out_neighbor_distance)

def find_neighbor_response (row_neighbor, response):
    row_response = []
    for index, row in row_neighbor.iterrows():
        row_response.append(response.loc[row].values)
    out_neighbor_response = pandas.DataFrame(row_response).set_index(row_neighbor.index)
    return(out_neighbor_response)

toyData = pandas.DataFrame([[7.7, -37, 4], [9.5, -38, 1], [3,  -34, 2], [9.1, -75, 1],
                            [2.2, -31, 2], [4.8,  -7, 4], [5.5, -6, 3], [10,  -61, 1],
                            [4.2, -23, 2], [1.6, -54, 1]],
                           index = ['A','B','C','D','E','F','G','H','I','J'],
                           columns = ['x1','x2','y'])

# Visualize the relative positions of the points (x1,x2)
plt.figure(figsize = (8,6), dpi = 200)
plt.scatter(toyData['x1'], toyData['x2'], marker = 'o')
for index, row in toyData.iterrows():
    plt.annotate(index, (row['x1']+0.1, row['x2']+0.1))
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(axis = 'both')
plt.show()

# Train the Nearest Neighbor using Euclidean distance
pairwise_euclidean, row_rank = pair_euclidean(toyData[['x1','x2']])

# Specifiy the number of neighbors
n_neighbor = 10

# Find the neighbors for each row
row_neighbor = find_neighbor(row_rank, n_neighbor)

# Find distances from each row to its neighbors
row_neighbor_distance = find_neighbor_distance(pairwise_euclidean, row_neighbor)

# Find the neighbors' responses for each row
row_neighbor_response = find_neighbor_response(row_neighbor, toyData['y'])

# Calculate the mean of the neighbors' response
mean_response = row_neighbor_response.mean(axis = 1)

# Calculate the prediction error
error_response = toyData['y'] - mean_response

# Calculate the sum of squares error
sse = numpy.sum(numpy.power(error_response, 2))