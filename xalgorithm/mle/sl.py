__all__ = ['KNN', 'MLP']
"""Supervised Learning

- KNN: k nearest neighbor classifier
"""
import numpy as np
from .utils import eucliean_distance, get_progress
from .loss import *
from .activate import *

class Base:
    @property
    def name(self): return self.__class__.__name__
    def __call__(self, X):
        raise NotImplementedError("The predicton function should be implemented")
    def fit(self, X, y, **kwargs):
        raise NotImplementedError("The model fit function should be implemented")

class KNN(Base):
    r"""KNN does not involve a traditional training process where model
    parameters are learnt from the data

    Instead, it memorizes the training data and makes predictions based on the proximity of new data points to the stored examples.
    
    ```
    >>> m = KNN(5)
    >>> m.fit(Xtrain, ytrain) # {X: (B, E), y: (B, )}
    >>> m(Xtest)
    ```
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train, self.y_train = X, y

    def _vote(self, indices: np.ndarray): 
        count = np.bincount(self.y_train[indices].astype(int))
        return count.argmax() 
    
    def __call__(self, X):
        if self.X_train is None or self.y_train is None:
            raise ValueError("The model has not been trained. Please call 'fit' first.")
        pred = np.empty(X.shape[0])
        for i, X_test in enumerate(X):
            """return k indices with lowest distance to this instance """
            dist = [eucliean_distance(X_train, X_test) for X_train in self.X_train]
            k_idx = np.argsort(dist)[:self.k]
            pred[i] = self._vote(k_idx) 
        return pred

        
# The MLP class is a subclass of the Base class and represents a multi-layer perceptron with
# adjustable learning rate, loss function, and activation function.
class MLP(Base):
    r"""A multi-layer perceptron with adjustable learning rate, loss function, and activation function

    ```
    >>> m = MLP(lr=1e-3, loss=CrossEntropy(), nonlinear=Sigmoid())
    >>> m.fit(Xtrain, ytrain) # {X: (B, E), y: (B, O)}
    >>> m(Xtest)
    ```
    """
    def __init__(self, loss=None, lr=1e-2, nonlinear=None):
        self.lr = lr
        self.loss = loss
        self.act  = nonlinear
        self.k, self.b = None, None

    def __call__(self, X):
        r"""
        :eq:    y = σ(k'X ⋅ b) 
        :rtn:   y ∈ R(1, N)
        """
        return self.act(X.dot(self.k) + self.b)

    def fit(self, X, y, **kwargs):
        r""" Takes in input data X and labels y, and fits a linear line with slope {k} and intercept {b}
        
        @param X        The input data matrix with shape (n_samples, n_features).
        @param y        The target variable or the dependent variable in the dataset.
        @param n_iter   The number of epochs to train at most (default=20_000)
        @param lr       The learning rate of this algorithm (default=self.lr)
        @param tol      The tolerate value where the change of loss is lower than this number, the algorithm stops training

        @weight k       The slope parameter, initialized between -1/sqrt(N) ~ 1/sqrt(N)
        @weight b       The intercept, initialized with zero
        """
        n_iter = kwargs.get('n_iter', 20_000)
        lr = kwargs.get('lr', self.lr)
        tol = kwargs.get('tol', None)
        if np.ndim(y) == 2:
            (_, n_features), (_, n_outputs) = map(np.shape, (X, y))
        elif np.ndim(y) == 1:
            (_, n_features), n_outputs = np.shape(X), 1
            y = y.reshape(-1, 1)
        LB, UB = -1/np.sqrt(n_features), 1/np.sqrt(n_features)
        self.k = np.random.uniform(LB, UB, (n_features, n_outputs))
        self.b = np.zeros((1, n_outputs))
        # with get_progress() as p:
            # task = p.add_task("[cyan]Training...", total=n_iter)
        for epoch in range(n_iter):
            out = X.dot(self.k) + self.b
            pred = self.act(out)
            grad = self.loss(y, pred, grad=True) * self.act(out, grad=True)
            # Calculate the gradient of the loss with respect to k and b
            grad_k = X.T.dot(grad)
            grad_b = np.sum(grad, axis=0, keepdims=True)
            # Update coefficients
            self.k -= lr * grad_k
            self.b -= lr * grad_b
            loss = np.mean(self.loss(y, pred))
            if epoch % 1000 == 0: 
                # p.update(task, completed=epoch, description=f"Epoch {epoch}, Loss {loss:.4f}\n")
                print(f"Epoch {epoch}, Loss {loss:.4f}\n")
            if tol and loss < tol: break

