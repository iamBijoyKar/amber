import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __repr__(self):
        return str(f'<amber.Tensor dtype={self.data.dtype} array=(\n{self.data}\n) >')