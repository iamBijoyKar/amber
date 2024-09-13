import numpy as np
from .utils import generate_weights, generate_bias
from .nurons import Neuron

class InputLayer:
    def __init__(self,nurons:int):
        self.nurons = []
        self.output_length = nurons
        for _ in range(nurons):
            new_nuron = Neuron() 
            self.nurons.append(new_nuron)
        
    def __repr__(self) -> str:
        return f'<amber.Layers.InputLayer nurons={len(self.nurons)} >'
    
    def forward(self,inputs:list[float]):
        if len(inputs) != len(self.nurons):
            raise ValueError(f'Inappropriate input size given at {self.__repr__} layer. Given size is {np.array(inputs).shape} but required is ({len(self.nurons)})')
        outputs = []
        for i in range(len(inputs)):
            outputs.append(self.nurons[i].forward(inputs,idx=i))
        return np.array(outputs)
    

class Dense:
    def __init__(self,nurons:int):
        self.nurons = []
        self.output_length = nurons
        self.input_length = None

    def compile(self,input_length):
        self.input_length = input_length
        for _ in range(self.output_length):
            weights = generate_weights(self.input_length)
            bias = generate_bias()
            new_nuron = Neuron(weights,bias) 
            self.nurons.append(new_nuron)


    def __repr__(self) -> str:
        return f'<amber.Layers.Dense nurons={len(self.nurons)} >'
    
    def forward(self,inputs:list[float]):
        if self.input_length == None:
            raise RuntimeError(f'The model {self.__repr__} is uncompiled, compile it before run.')
        outputs = []
        for i in range(len(self.nurons)):
            outputs.append(self.nurons[i].forward(inputs))
        print(outputs)
        return np.array(outputs)