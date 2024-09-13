import random
import numpy as np

def generate_weights(num:int):
    weights = []
    for _ in range(num):
        weights.append(random.random()*random.choice([-1,1]))
    return np.array(weights)

def generate_bias():
    return random.random()*random.choice([-1,1])
