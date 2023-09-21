__all__ = ['accuracy', 'confusion_matrix', 'precision', 'recall', 'f1_score']

import numpy as np
# from numba import jit

def check_valid(actual, predicted):
    if isinstance(actual, list): actual = np.array(actual)
    if isinstance(predicted, list): predicted = np.array(predicted)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("actual and predicted must be 1D arrays")
    if actual.size == 0 or predicted.size == 0:
        raise ValueError("actual and predicted must not be empty")
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    return (actual, predicted)

# ================================ Classification =================================== #
def accuracy(actual, predicted):
    """ Accuracy  = (TP+TN) / ALL """
    actual, predicted = check_valid(actual, predicted)
    return np.mean(actual == predicted)

def confusion_matrix(actual, predicted):
    """ Confusion Metrics 

           N'   \tP' \n
        N  \tTN    FP \n
        P  \tFN    TP
    """
    actual, predicted = check_valid(actual, predicted)
    classes = np.unique(actual)
    classes = np.unique(actual)
    metrics = np.zeros((classes.size, classes.size))
    for i, c in enumerate(classes):
        for j, d in enumerate(classes):
            metrics[i, j] = np.sum((actual == c) & (predicted == d))
    return metrics
    

def precision(actual, predicted):
    """ Precision = TP / P' """
    actual, predicted = check_valid(actual, predicted)
    metrics = confusion_matrix(actual, predicted)
    return metrics[1, 1] / np.sum(metrics[:, 1])

def recall(actual, predicted):
    """ Recall = TP / P """
    actual, predicted = check_valid(actual, predicted)
    metrics = confusion_matrix(actual, predicted)
    return metrics[1, 1] / np.sum(metrics[1, :])

def f1_score(actual, predicted):
    r""" 
        F1 =  2 x (Pre x Rec) / (Pre + Rec)
           =  2 x TP / (P + P')
    """
    actual, predicted = check_valid(actual, predicted)
    metrics = confusion_matrix(actual, predicted)
    return 2 * metrics[1,1] / (sum(metrics[1, :] + metrics[:, 1]))