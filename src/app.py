from lib.nn import Net
from random import seed
import csv
import sys
import numpy as np

data_path = '../'

X = np.genfromtxt(data_path + 'X.csv', delimiter=',')
Y_plain = Y = np.genfromtxt(data_path + 'Y.csv', delimiter=',')
Y = np.vstack(Y_plain)
dataset = np.concatenate([X, Y], axis = 1)

n_inputs = len(dataset[0]) - 1
print("INPUT DIM ")
print(n_inputs)

n_outputs = len(set([row[-1] for row in dataset]))
print("OUTPUT DIM ")
print(n_outputs)

seed(1)
shape = {
    "inputs": n_inputs,
    "hidden": 6,
    "outputs": n_outputs
}
net = Net(shape) #inicializa la red

net.train_network(dataset, 0.040, 100, n_outputs) #1821 de 20000
print(net)

j = 0
for row in dataset:
    prediction = net.predict(row)
    if(row[-1]!=prediction):
        j = j+1
    #print('Expected=%d, Got=%d' % (row[-1], prediction))

print(j)

