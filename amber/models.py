from .layers import InputLayer, Dense

class Model:
    def __init__(self,layers:list):
        for i in range(len(layers)):
            if i == 0:
                if not isinstance(layers[i],InputLayer):
                    raise TypeError(f'The first layer should be a <amber.layers.InputLayer> but recieved {type(layers[i])}')
            else:
                if not isinstance(layers[i],Dense):
                    raise TypeError(f'The first layer should be a <amber.layers.Dense> but recieved {type(layers[i])}')
        self.layers = layers

    def forward(self,inputs:list[float]):
        new_input = inputs
        output = []
        for layer in self.layers:
            output = layer.forward(new_input)
            new_input = output
        
        return new_input
            



    def __repr__(self) -> str:
        return f'<amber.models.Model layer={len(self.layers)}>'

