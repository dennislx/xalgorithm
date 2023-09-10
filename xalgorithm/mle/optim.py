__all__ = ['SGD', 'Optim']

import numpy as np
from .utils import or_get_default

class Optim:
    def __init__(self, lr): self.lr = lr
    def update(self, w, grad_wrt_w): raise NotImplementedError

class SGD(Optim):
    """Stochastic Gradient Descent

        velocity = eta x velocity + (1-eta) x grad \n
        w' = w - lr * velocity
    """
    def __init__(self, lr: float =1e-2, momentum:float =0):
        super().__init__(lr)
        self.eta = momentum
        self.velocity = None
    def update(self, w, grad_wrt_w):
        v = or_get_default(self.velocity, np.zeros(np.shape(w)))
        self.velocity = self.eta * v + (1-self.eta) * grad_wrt_w
        return w - self.lr * self.velocity


        
