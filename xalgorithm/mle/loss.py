__all__ = ['MSE', 'CrossEntropy']

from typing import Any
import numpy as np

class Loss:
    def __call__(self, y_true, y_pred, grad):
        raise NotImplementedError()
    def grad(self, y_true, y_pred):
        raise NotImplementedError()

class MSE(Loss):
    def __call__(self, y_true, y_pred, grad=False):
        """ L = Sum (x - y)^2 """
        if grad: return self.grad(y_true, y_pred)
        return 0.5 * np.power((y_true - y_pred), 2)
    def grad(self, y_true, y_pred):
        return -(y_true - y_pred)

class CrossEntropy(Loss):
    def __init__(self, eta=1e-15): 
        r""" The above function initializes an object with an eta value and a clip function that clips values between eta and 1-eta.
        
        @param eta The parameter `eta` is a small positive value used to avoid taking the logarithm of zero
        """
        self.eta = eta
        self.clip = lambda _: np.clip(_, eta, 1-eta)
    def __call__(self, y_true, y_prob, grad=False):
        r""" The function calculates the cross-entropy (neg loglikelihood?) loss between the true labels and predicted probabilities.

        :Eq:          - { ylogp + (1-y)log(1-p) }
        """
        if grad: return self.grad(y_true, y_prob)
        y_prob = self.clip(y_prob)
        return -y_true*np.log(y_prob) - (1-y_true)*np.log(1-y_prob)

    def grad(self, y_true, y_prob):
        y_prob = self.clip(y_prob)
        return - (y_true / y_prob) + (1-y_true) / (1-y_prob)

if __name__ == '__main__':
    pass