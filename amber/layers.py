import numpy as np
from .utils import generate_weights, generate_bias
from .neurons import Neuron

class InputLayer:
    def __init__(self,input_length:int):
        self.input_length = input_length
        self.neurons = []
        self.output_length = input_length
        for _ in range(input_length):
            new_neuron = Neuron() 
            self.neurons.append(new_neuron)
        
    def __repr__(self) -> str:
        return f'<amber.Layers.InputLayer neurons={len(self.neurons)} >'
    
    def forward(self,inputs:list[float]):
        if len(inputs) != len(self.neurons):
            raise ValueError(f'Inappropriate input size given at {self.__repr__} layer. Given size is {np.array(inputs).shape} but required is ({len(self.neurons)})')
        outputs = []
        for i in range(len(inputs)):
            outputs.append(self.neurons[i].forward(inputs,idx=i))
        return np.array(outputs)
    

class Dense:
    def __init__(self,neurons:int,activation=None):
        self.neurons = []
        self.output_length = neurons
        self.input_length = None
        self.activation = activation

    def compile(self,input_length):
        self.input_length = input_length
        if self.activation != 'relu' and self.activation != 'sigmoid':
            raise ValueError(f'Invalid activation function. Given {self.activation} but expected `relu` or `sigmoid`.')
        for _ in range(self.output_length):
            weights = generate_weights(self.input_length)
            bias = generate_bias()
            new_neuron = Neuron(weights,bias,activation=self.activation) 
            self.neurons.append(new_neuron)


    def __repr__(self) -> str:
        return f'<amber.Layers.Dense neurons={len(self.neurons)} >'
    
    def forward(self,inputs:list[float]):
        if self.input_length == None:
            raise RuntimeError(f'The model {self.__repr__} is uncompiled, compile it before run.')
        outputs = []
        for i in range(len(self.neurons)):
            outputs.append(self.neurons[i].forward(inputs))

        return np.array(outputs)
    

class SoftMax:
    def __init__(self) -> None:
        self.input_length = None
        self.output_length = None

    def compile(self,input_length):
        self.input_length = input_length
        self.output_length = input_length

    def forward(self,inputs:list[float]):
        if self.input_length == None:
            raise RuntimeError(f'The model {self.__repr__} is uncompiled, compile it before run.')
        inputs_sum = np.sum(inputs)
        output = [i / inputs_sum if inputs_sum != 0 and i != 0 else 1 for i in inputs]
        return np.array(output)