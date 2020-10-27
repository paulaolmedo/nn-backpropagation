from lib.nn import Net
from random import seed
import csv
import sys
import numpy as np
"""
dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]]

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
seed(1)
shape = {
    "inputs": 2,
    "hidden": 2,
    "outputs": 2
}
net = Net(shape) #inicializa la red
print(net)
net.train_network(dataset, 0.5, 20, n_outputs)
print(net)

for row in dataset:
    prediction = net.predict(row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))

"""
#define x_data
x_data = open('/Users/paulabolmedo/Documents/Facultad/X.csv')
x_coordinates = csv.reader(x_data, delimiter=',')
#define y_data
y_data = open('/Users/paulabolmedo/Documents/Facultad/Y.csv')
y_coordinates = csv.reader(y_data, delimiter=',')
    
x_list = []
y_list = []
    
for row in x_coordinates:
    x_list.append(row)
    
for row in y_coordinates:
    y_list.append(row)
    

new_x_list = []
for sub in x_list:
    new_sub = []
    for item in sub:
        new_item = float(item)
        new_sub.append(new_item)
    new_x_list.append(new_sub)

x_array = np.array(new_x_list)

new_y_list = []
for sub in y_list:
    new_sub = []
    for item in sub:
        new_item = float(item)
        new_sub.append(int(new_item))
    new_y_list.append(new_sub)

y_array = np.array(new_y_list)

dataset = np.concatenate([x_array, y_array], axis = 1)

n_inputs = len(dataset[0]) - 1
print("INPUT DIM ")
print(n_inputs)

n_outputs = len(set([row[-1] for row in dataset]))
print("OUTPUT DIM ")
print(n_outputs)

seed(1)
shape = {
    "inputs": n_inputs,
    "hidden": 3,
    "outputs": n_outputs
}
net = Net(shape) #inicializa la red

net.train_network(dataset, 0.009, 50, n_outputs)
print(net)

j = 0
for row in dataset:
    prediction = net.predict(row)
    if(row[-1]!=prediction):
        j = j+1
    #print('Expected=%d, Got=%d' % (row[-1], prediction))

print(j)

