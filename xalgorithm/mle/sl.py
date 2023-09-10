__all__ = ['KNN']
"""Supervised Learning

- KNN: k nearest neighbor classifier
"""
import numpy as np
from .utils import eucliean_distance

class Base:
    @property
    def name(self): return self.__class__.__name__

class KNN(Base):
    r"""KNN does not involve a traditional training process where model
    parameters are learnt from the data

    Instead, it memorizes the training data and makes predictions based on the proximity of new data points to the stored examples.
    
    ```
    >>> m = KNN(5)
    >>> m.fit(Xtrain, ytrain) # {X: (B, E), y: (B, )}
    >>> m.predict(Xtest)
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
    
    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise ValueError("The model has not been trained. Please call 'fit' first.")
        pred = np.empty(X.shape[0])
        for i, X_test in enumerate(X):
            """return k indices with lowest distance to this instance """
            dist = [eucliean_distance(X_train, X_test) for X_train in self.X_train]
            k_idx = np.argsort(dist)[:self.k]
            pred[i] = self._vote(k_idx) 
        return pred

            
        
        