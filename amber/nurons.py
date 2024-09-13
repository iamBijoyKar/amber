import numpy as np
from .activation import relu,sigmoid
from .utils import generate_bias

class Neuron:
    def __init__(self, weights=[], bias=generate_bias(),activation=None):
        self.type = "input" if len(weights) == 0 else "dense"
        self.weights = np.array(weights)
        self.bias = bias
        self.activation = activation

    def forward(self, inputs:list[float],idx=0):
        if self.type == 'input':
            return inputs[idx]
        prod = np.sum(np.dot(inputs, self.weights))
        if self.activation == None:
            return prod + self.bias
        elif self.activation == 'relu':
            return relu(prod + self.bias)
        elif self.activation == 'sigmoid':
            return sigmoid(prod + self.bias)
        else: 
            raise ValueError(f'Invalid activation function. Given {self.activation} but expected `relu` or `sigmoid`.')

    def __repr__(self):
        return str(f'<amber.Neuron type={self.type} weights={self.weights} bias={self.bias} >')
    


