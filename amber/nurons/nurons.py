import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def __repr__(self):
        return str(f'<amber.Neuron weights={self.weights} bias={self.bias} >')
