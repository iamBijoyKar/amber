from .layers import InputLayer, Dense, SoftMax
from .loss import categorical_cross_entropy

class Model:
    def __init__(self,layers:list):
        for i in range(len(layers)):
            if i == 0:
                if not isinstance(layers[i],InputLayer):
                    raise TypeError(f'The first layer should be a <amber.layers.InputLayer> but recieved {type(layers[i])}')
            else:
                if not isinstance(layers[i],Dense) and not isinstance(layers[i],SoftMax) :
                    raise TypeError(f'The first layer should be a <amber.layers.Dense> but recieved {type(layers[i])}')
        self.layers = layers
        self.inputs = []
        self.outputs = []
        self.true_outputs = []
        self.loss = None


    def compile(self):
        output_length = self.layers[0].output_length
        for i in range(1,len(self.layers)):
            self.layers[i].compile(output_length)
            output_length = self.layers[i].output_length

    def forward(self,inputs:list[float]):
        self.inputs = inputs # updating the inputs 
        new_input = inputs
        output = []
        for layer in self.layers:
            output = layer.forward(new_input)
            new_input = output
        self.outputs = new_input # updating the outpush
        return new_input
    
    def calc_loss(self,true_outputs:list[float]):
        if len(true_outputs) != len(self.outputs):
            raise ValueError(f'Invalid shape recieved! Given ({len(true_outputs)},) , expected ({len(self.outputs)},)')
        self.true_outputs = true_outputs
        loss =  categorical_cross_entropy(self.outputs,self.true_outputs)
        self.loss = loss
        return loss
    
    def fit(self,X:list[float],y:list[float],epoch=1):
        for _ in range(epoch):
            self.forward(X)
            self.calc_loss(y)
            print(f'Loss : {self.loss}')



    def __repr__(self) -> str:
        return f'<amber.models.Model layer={len(self.layers)}>'

