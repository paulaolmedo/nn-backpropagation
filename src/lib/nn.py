import numpy as np
import math
import csv
import pickle
import matplotlib.pyplot as plt
import sys

from random import random
from math import exp

# Transfer neuron activation
def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

def sigmoid_derivative(output):
    return output * (1.0 - output)

def tanh(activation):
    return np.tanh(activation)

def tanh_derivative(output):
    return 1.0 - (output*output)

class Neuron:
    def __init__(self, weight=None):
        if not weight:
            self.weight = random()
        else:
            self.weight = weight
            
    def __str__(self):
        return str(self.weight)
    


class Layer:
    def __init__(self, n, neurons=None):
        if not neurons:
            self.neurons = []
            for i in range(n):
                self.neurons.append(Neuron())
        else:
            self.neurons = neurons
        
        #self.bias = Neuron()
        self.output = None 
        self.delta = None 
        
    def __str__(self):
        #print(str(self.neurons))
        neurons_print = "neurons = ["
        for neuron in self.neurons:
            neurons_print = neurons_print + str(neuron) + ", "
        neurons_print = neurons_print[:-2] + "]\n"
        #bias_print = "bias = {} \n".format(self.bias)
        output_print = "output = {}\n".format(self.output)
        delta_print = "delta = {}\n".format(self.delta)
        
        #return neurons_print + bias_print + output_print + delta_print
        return neurons_print + output_print + delta_print
        
                
    def activate(self, layer, inputs):
        activation = layer[-1].weight #self.bias.weight
        for i in range(len(layer) -1):
            activation += layer[i].weight * inputs[i]
        return activation
    

            
class Net:
    def __init__(self, shape):
        """
        arguments: shape is going to be an array indicating the
                   size of each layer, eg [2, 1, 2] asks for a
                   network with 2 input neurons, one hidden and
                   2 outputs.
        """
        self.layers = list()
        self._init_layers(shape)

        
    def _init_layers(self, shape):
        self.hidden_layer = []
        for i in range(shape["hidden"]):
            #print("created %s hidden", shape["hidden"])
            self.hidden_layer.append(Layer(shape["inputs"] + 1))
        
        self.output_layer = []
        for i in range(shape["outputs"]):
            #print("created %s out", shape["outputs"])
            self.output_layer.append(Layer(shape["hidden"] + 1))
    
        self.layers.append(self.hidden_layer)
        self.layers.append(self.output_layer)
          
        
    def __str__(self):
        out = ""
        for super_layer in self.layers:
            out = out + "----super layer----\n"
            for sub_layer in super_layer:
                out = out + "--sub layer--\n"
                out = out + str(sub_layer)
        return out
    
    def forward_propagate(self, net_inputs):
        inputs = net_inputs
        for super_layer in self.layers:
            temporal_output = []
            for sub_layer in super_layer:
                activation = sub_layer.activate(sub_layer.neurons, inputs)
                output = tanh(activation) #sigmoid(activation) #tanh(activation)
                sub_layer.output = output
                temporal_output.append(output)

            inputs = temporal_output
        return inputs
    
    def backward_propagate_error(self, expected):
        net_range = len(self.layers)
        #print("net_range: ")
        #print(net_range)
        
        for i in reversed(range(net_range)):
            layer = self.layers[i]
            errors = list()
            
            layer_range = len(layer)
            #print("layer_range: ")
            #print(layer_range)
            
            if i != (net_range - 1):         
                for j in range(layer_range):
                    error = 0.0
                    for element in self.layers[i+1]:
                        error += element.neurons[j].weight * element.delta
                    errors.append(error)
            else:
                for j in range(layer_range):
                    element = layer[j]
                    output = element.output
                    errors.append(expected[j] - output)
                
            for j in range(layer_range):
                    element = layer[j]
                    element.delta = errors[j] * tanh_derivative(element.output) #sigmoid_derivative(element.output) 
    
    def update_weights(self, net_inputs, learning_rate):
        net_range = len(self.layers)
        
        for i in range(net_range):            
            inputs = net_inputs[:-1]

            if i != 0:
                inputs = [element.output for element in self.layers[i-1]]
                
            for element in self.layers[i]:
                for j in range(len(inputs)):
                    element.neurons[j].weight += learning_rate * element.delta * inputs[j]
                element.neurons[-1].weight += learning_rate * element.delta

    def train_network(self, dataset, learnig_rate, epochs, n_outputs):
        for epoch in range(epochs):
            sum_error = 0
            for row in dataset:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(n_outputs)]
                expected[int(row[-1])] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, learnig_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learnig_rate, sum_error))
    
    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))