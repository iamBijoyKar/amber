import numpy as np

def binary_cross_entropy(p:list,q:list) -> float:
    return -np.sum([q[i]*np.log2(p[i]) + (1-q[i])*np.log2(1-p[i]) for i in range(len(q))])

def categorical_cross_entropy(p:list,q:list) -> float:
    return -np.sum([q[i]*np.log2(p[i]) for i in range(len(q))])