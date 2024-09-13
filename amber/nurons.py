import numpy as np
from .utils import generate_bias

class Neuron:
    def __init__(self, weights=[], bias=generate_bias()):
        self.type = "input" if len(weights) == 0 else "dense"
        self.weights = np.array(weights)
        self.bias = bias

    def forward(self, inputs:list[float],idx=0):
        if self.type == 'input':
            return inputs[idx]
        prod = np.sum(np.dot(inputs, self.weights))
        return prod + self.bias

    def __repr__(self):
        return str(f'<amber.Neuron type={self.type} weights={self.weights} bias={self.bias} >')
    


