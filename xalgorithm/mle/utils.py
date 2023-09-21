__all__ = [
    "train_test_split", 
    "or_get_default", 
    "eucliean_distance",
    "gen_sample_data",
    "get_progress",
]
import numpy as np
from sklearn import datasets
from rich.progress import *

def get_progress():
    text_column = TextColumn("[progress.description]{task.description}", table_column=Column(ratio=2))
    remain_column = TimeRemainingColumn(compact=True, table_column=Column(ratio=1))
    bar_column = BarColumn(bar_width=None, table_column=Column(ratio=2))
    percentage_column = TaskProgressColumn()
    p = Progress(
        text_column, remain_column, bar_column, percentage_column
    )
    return p

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    y_true = y_true.reshape(y_pred.shape)
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def assert_1d_array(arr):
    if isinstance(arr, np.ndarray):
        assert arr.ndim == 1, "Array is not one-dimensional"
    elif isinstance(arr, list):
        assert all(isinstance(x, (int, float, str)) for x in arr), "List is not one-dimensional"

def eucliean_distance(x1, x2):
    assert_1d_array(x1)
    assert_1d_array(x2)
    return np.linalg.norm(x1 - x2, axis=0)

def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed: np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def or_get_default(x, default):
    return default if x is None else x

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle: X, y = shuffle_data(X, y, seed)
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]
    return X_train, X_test, y_train, y_test

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def gen_sample_data(purpose='binaryclass'):
    """ Generalize sample dataset """
    data = datasets.load_digits()
    target: list = data.target # type: ignore
    if purpose == 'binaryclass':
        idx = np.append( np.where(target == 1)[0], np.where(target == 8)[0])
        X, y = data.data[idx], target[idx] # type: ignore
        y[y==1] = 0; y[y==8] = 1;
    elif purpose == 'multiclass':
        X = data.data
        y = to_categorical(target)
    return normalize(X), y

def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col: n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

    
if __name__ == '__main__':
    import time
    with get_progress() as p:
        task = p.add_task("[red]Downloading...", total=1000)
        while not p.finished:
            p.update(task, advance=0.5)
            time.sleep(0.1)