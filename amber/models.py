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


    def compile(self):
        output_length = self.layers[0].output_length
        for i in range(1,len(self.layers)):
            self.layers[i].compile(output_length)
            output_length = self.layers[i].output_length

    def forward(self,inputs:list[float]):
        new_input = inputs
        output = []
        for layer in self.layers:
            output = layer.forward(new_input)
            new_input = output
        
        return new_input
            



    def __repr__(self) -> str:
        return f'<amber.models.Model layer={len(self.layers)}>'

