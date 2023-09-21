__all__ = ['Sigmoid', 'ReLU']

import numpy as np

class Activate:
    def __call__(self, x, grad):
        raise NotImplementedError()
    def grad(self, x):
        raise NotImplementedError()

class Sigmoid(Activate):
    def __call__(self, x, grad=False):
        """ 1 / (1 + e^x) """
        if grad: return self.grad(x)
        return 1 / (1 + np.exp(-x))
    def grad(self, x):
        return self(x) * (1-self(x))

class ReLU(Activate):
    def __call__(self, x, grad=False):
        """max(0, x)"""
        if grad: return self.grad(x)
        return np.where(x > 0, x, 0)
    def grad(self, x):
        return np.where(x > 0, 1, 0)

if __name__ == '__main__':
    pass