import numpy as np
import warnings
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib import style
from collections import Counter
import random
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')

def k_nearest_neig( data, predict, k):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')

    distances = []

    for group in data:
        for features in data[group]:
            euclidean_dis = np.linalg.norm(np.array(features) - np.array(predict))
            #linalg mean linear algebra library and norm use for all othe sqrt and other calcuations
            distances.append([euclidean_dis, group])

    vote = [ j[1] for j in sorted(distances)[:k]]
    vote_result = Counter(vote).most_common(1)[0][0]

    return vote_result   #it will return the group of the newly classified data



#Main program

dataset = pd.read_csv("breast-cancer-wisconsin.data.txt")
dataset.drop(dataset.columns[[0]], 1, inplace=True)
dataset.replace('?', -9999, inplace=True)
#print(dataset)
full_data = dataset.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.5    #testing 40% of data
train_set = { 2:[], 4:[] }
test_set = { 2:[], 4:[] }

train_data = full_data[ :int(test_size * len(full_data)) ]
test_data = full_data[ int(test_size * len(full_data)): ]

# -1 is the last column in data and last column shows the class for data so thats why
#first of all we are selecting the class of data then appending to the last column
for row_data in train_data:
    train_set[row_data[-1]].append(row_data[:-1])

for row_data in test_data:
    test_set[row_data[-1]].append(row_data[:-1])

correct = 0
total = 0

set = { 2 : '#10319e', 4: '#9e1010' }

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neig( train_set, data, k=7)
        if vote == group:
            correct += 1
        ax.scatter(data[0], data[1], data[2], s=20, color=set[vote])
        total += 1

ax.set_xlabel('Clump Thickness')
ax.set_ylabel('Uniformity of Cell Size')
ax.set_zlabel('Uniformity of Cell Shape')


plt.legend(['Benign Tumor', 'Malignant Tumor'], loc=2)
az = plt.gca()
leg = az.get_legend()
leg.legendHandles[0].set_color('#10319e')
leg.legendHandles[1].set_color('#9e1010')

acc = (correct / total) * 100
print('Accuracy:', acc, '%')

plt.show()
