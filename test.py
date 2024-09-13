from amber.layers import InputLayer, Dense
from amber.models import Model

input_layer = InputLayer(8)
dense_1 = Dense(10,input_length=8)

model = Model([input_layer,dense_1])

res = model.forward([1,2,3,4,5,6,7,8])

print(len(res))

