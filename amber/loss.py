import numpy as np

EPSILON = 1e-10

def binary_cross_entropy(p:list,q:list) -> float:
    return -np.sum([max(q[i], EPSILON)*np.log2(p[i] + EPSILON) + max((1-q[i]), EPSILON)*np.log2(1 - p[i] + EPSILON) for i in range(len(q))])

def categorical_cross_entropy(p:list,q:list) -> float:
    return -np.sum([max(q[i], EPSILON)*np.log2(p[i] + EPSILON) for i in range(len(q))])

def square_cost(p:list,q:list)->float:
    return np.sum([[(p[i] - q[i])**2 for i in range(len(p))]])
