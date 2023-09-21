__all__ = ['Plot']

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
from xalgorithm import ojoin, osimplify

DEFAULT_PATH = osimplify(ojoin(__file__, '../../..', 'figs'))

def get_correlation(X, Y=None):
    if Y is None: Y = X
    N = np.shape(X)[0]
    Xbar, Ybar = map(lambda _: np.mean(_, axis=0), (X, Y))
    covariance = (1 / N) * (X - Xbar).T.dot(Y - Ybar)
    Xstd, Ystd = map(lambda _: np.expand_dims(np.std(_, axis=0),1), (X, Y))
    corr = np.divide(covariance, Xstd.dot(Ystd.T))
    return np.array(corr, dtype=float)

def get_covariance(X, Y=None):
    if Y is None: Y = X
    N = np.shape(X)[0]
    Xbar, Ybar = map(lambda _: np.mean(_, axis=0), (X, Y))
    covar = (1 / (N-1)) * (X - Xbar).T.dot(Y - Ybar)
    return np.array(covar, dtype=float)

class Plot:
    def __init__(self, path=DEFAULT_PATH, title=None):
        self.cmap = plt.get_cmap('viridis')
        self.title = title
        self.path = path

    def _transform(self, X, dim):
        covariance = get_covariance(X)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        idx = eigenvalues.argsort()[::-1] # get most {dim} largest eigen
        eigenvalues = eigenvalues[idx][:dim]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :dim]
        return X.dot(eigenvectors)

    def plot2d(self, X, y, name='2dplot', legend_labels=None):
        X = self._transform(X, dim=2)
        x1, x2 = X[:, 0], X[:, 1]
        y = np.array(y.reshape(-1,)).astype(int)
        palette = sns.color_palette("husl", len(np.unique(y)))
        ps = []
        fig = plt.figure(figsize=(8, 6))
        for label in np.unique(y):
            x11, x22 = x1[y == label], x2[y == label]
            ps.append(plt.scatter(x11, x22, label=f'Class {label}', color=palette[label]))
        if not legend_labels is None: 
            plt.legend(ps, legend_labels, loc=1)
        self.title and plt.title(self.title)
        plt.xlabel('X1'); plt.ylabel('X2')
        fig.savefig(ojoin(self.path, '{}.png'.format(name)))
        



        