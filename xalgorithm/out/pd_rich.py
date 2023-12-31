__all__ = ['print_df', 'describe_df', 'chi2_test', 'Benchmark', 'plot_df']

"""Modified from [rich-dataframe](https://pypi.org/project/rich-dataframe/)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.table import Table
from rich import print as rprint
from scipy import stats
from sklearn.datasets import make_classification as MC
from sklearn.metrics import get_scorer
from sklearn.utils import all_estimators
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import random
import inspect
import time
import re

COLORS = ["cyan", "magenta", "red", "green", "blue", "purple"]
NCOLOR = len(COLORS)

DEFAULT_METRICS = {
    'acc': 'accuracy',
    'prec': 'precision',
    'recall': 'recall',
    'f1-macro': 'f1_macro',
    'f1-micro': 'f1_micro',
    'f1-weight': 'f1_weighted',
    'logloss': 'neg_log_loss',
    'auc': 'roc_auc'
}



SELECTED = [
    'AdaBoost', 'RandomForest', 'DecisionTree', 'GaussianProcess', 'GradientBoosting',
    'KNeighbors', 'MLP', 'PassiveAggressive', 'SGD', 'BernoulliNB','LogisticRegression',
]
ALLMODELS = all_estimators() 
get_model = lambda _: _.split('Classifier')[0] in SELECTED
CLASSIFIERS = {
    name: model for name, model in ALLMODELS
    if (get_model(name) and issubclass(model, ClassifierMixin))
}
CLASSIFIERS.update(XGBClassifier=XGBClassifier, LGBMClassifier=LGBMClassifier)

def get_arg_from_class(cls, method='__init__'):
    return set(inspect.signature(getattr(cls, method)).parameters)

def run_fit(X_train, y_train, X_test, y_test, model, metrics, model_name):
    start = time.time()
    model.fit(X_train, y_train)
    stat_metrics = {k: None for k in metrics}
    stat_metrics['time'] = time.time() - start
    y_pred, y_prob = model.predict(X_test), None
    if hasattr(model, 'predict_proba'): 
        y_prob = model.predict_proba(X_test)
    for name, metric in metrics.items():
        if y_prob is not None and name == 'logloss':
            stat_metrics[name] = metric(y_test, y_prob)
        elif y_prob is not None and name == 'auc':
            stat_metrics[name] = metric(y_test, y_prob[:, 1])
        else:
            stat_metrics[name] = metric(y_test, y_pred)
    return model_name, model, stat_metrics
    
def get_val_split(df, cols, n=11):
    cond = df[cols].nunique() > n
    card_high, card_low = cols[cond], cols[~cond]
    return card_low, card_high

def get_val_encoder():
    low = Pipeline(steps=[ ("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)), ] )
    high = Pipeline(steps=[ ("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("encoding", OrdinalEncoder()) ])
    return low, high


class Benchmark:
    
    r""" The function evaluates various widely recognized machine learning algorithms and offers several classification metrics for comparative analysis.

    ```
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn import datasets
    >>> from sklearn.utils import shuffle
    >>> boston = datasets.load_breast_cancer()
    >>> X, y = shuffle(boston.data, boston.target, random_state=13)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    >>> bench = benchmark_classification()
    >>> result = bench.fit(X_train, y_train, X_test, y_test)
    >>> fitted_models = bench.models
    ```
    """
    
    def __init__(self, models=CLASSIFIERS, metrics = DEFAULT_METRICS, random_seed=42, max_workers=None):
        r"""
        :param models:  A dictionary of name-classifier pairs that will be used for training and evaluation. 
        :param metrics: A dictionary of name-metric pairs that will be used to evaluate the performance of the selected models
        :param random_seed: To ensure the reproducibility of the benchmark result, set a fixed seed for randomness in algorithms (defaults to 42)
        :param max_workers: Specify the maximum number of worker processes to use for parallel processing. 
        """
        self.selected_models = models
        self.metrics = metrics
        self.seed = random_seed
        self.max_workers = max_workers
        self.prepare()
    
    @property
    def models(self): return self.fitted_models

    def prepare(self):
        def get_metric(score: str):
            scorer = get_scorer(score)
            f, kwargs = scorer._score_func, scorer._kwargs
            return lambda y_true, y_pred: f(y_true, y_pred, **kwargs)

        for name, metric in self.metrics.items():
            self.metrics[name] = get_metric(metric)
        self.fitted_models = None

    def add_model(self, **model_dict):
        for name, model in model_dict.items():
            self.selected_models[name] = model
    
    def add_model(self, **metric_dict):
        for name, metric in metric_dict.items():
            self.metrics[name] = metric

    def fit(self, X_train, y_train, X_test, y_test, **model_params):
        results = []
        if not isinstance(X_train, pd.DataFrame):
            X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        num_enc  = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scalar", StandardScaler())])

        cat_cols = X_train.select_dtypes(include=["object"]).columns
        cat_low_cols, cat_high_cols = get_val_split(X_train, cat_cols)
        cat_low_enc, cat_high_enc = get_val_encoder()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", num_enc, num_cols),
                ("categorical_low", cat_low_enc, cat_low_cols),
                ("categorical_high", cat_high_enc, cat_high_cols),
            ]
        )

        def get_model(name, model):
            model_args = model_params.get(name, {})
            if 'random_state' in get_arg_from_class(model):
                model_args.update(random_state=self.seed)
            model = model(**model_args)
            return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

        if self.max_workers is None:
            for name, model in tqdm(self.selected_models.items()):
                kwargs = dict( X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, metrics= self.metrics, model_name=name)
                results.append(run_fit(model=get_model(name, model), **kwargs))
        else:
            N = len(self.selected_models)
            jobs = [(X_train, y_train, X_test, y_test, get_model(x, y), self.metrics, x) for x,y in self.selected_models.items()]
            with ProcessPoolExecutor(max_workers=min(self.max_workers, N)) as executor:
                results = list(tqdm(executor.map(CloudpickleWrapper(run_fit), jobs, chunksize=1), total=N))
        index, model, metrics = zip(*results)
        self.fitted_models = model
        result = pd.DataFrame.from_records(metrics, index=index)
        print_df(result, sort_col='f1-weight')
        return result


class CloudpickleWrapper(object):
    r"""Uses cloudpickle to serialize contents (multiprocessing uses pickle by default)
    """
    def __init__(self, x):
        self.x = x
        
    def __call__(self, *args, **kwargs):
        return self.x(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.x, name)
    
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
        


def make_classification(n_samples, n_features, n_categories, weights=None):
    r"""Generates a synthetic classification dataset with specified number of samples, features, and categories.
    
    @param n_samples The number of samples or instances in the dataset.
    @param n_features The number of features in the dataset. Each feature represents a characteristic or
    attribute of the data points.
    @param n_categories The number of classes (or labels) of the classification problem.
    @param weights The proportions of samples assigned to each class. By default, it is set to `None`, which means that all categories have equal weights
    
    @return The function `make_classification` returns three values: `data`, `cat_cols`, and `num_cols`.
    ```
    >>> data, cat_cols, num_cols = make_classification(n_samples=10000, n_features=20, n_categories=4)
    ```
    """
    X,y = MC(n_samples=n_samples, n_features=n_features, random_state=42, n_informative=5, weights=weights)
    cat_cols = random.choices(list(range(X.shape[-1])),k=n_categories)
    num_cols = [i for i in range(X.shape[-1]) if i not in cat_cols]
    for col in cat_cols:
        X[:,col] = pd.qcut(X[:,col], q=4).codes.astype(int)
    col_names = [] 
    num_col_names=[]
    cat_col_names=[]
    for i in range(X.shape[-1]):
        if i in cat_cols:
            col_names.append(f"cat_col_{i}")
            cat_col_names.append(f"cat_col_{i}")
        if i in num_cols:
            col_names.append(f"num_col_{i}")
            num_col_names.append(f"num_col_{i}")
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y, name="target")
    data = X.join(y)
    return data, cat_col_names, num_col_names

def plot_df_helper(series: pd.Series, ax: plt.Axes, column: str, add_stat=True):
    hist_kws = dict(alpha=0.4, density=False)
    series = series.dropna()
    bins = min(sns.distributions._freedman_diaconis_bins(series), 50)
    ax.hist(series, bins, orientation='vertical', **hist_kws)
    if add_stat:
        text_kws = dict(transform=ax.transAxes, fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right')
        ax.text(x=0.97, y=0.96, s=f'Name: {column}', color='grey', **text_kws)
        ax.text(x=0.97, y=0.86, s=f'Skewness: {series.skew():5.3}', color='brown', **text_kws)
        ax.text(x=0.97, y=0.78, s=f'Kurtosis: {series.kurt():5.3}', color='green', **text_kws)
        ax.text(x=0.97, y=0.70, s=f'Minimum: {float(series.min()):5.3}', color='purple', **text_kws)
        ax.text(x=0.97, y=0.62, s=f'Maximum: {float(series.max()):5.3}', color='olive', **text_kws)

def plot_cv(df: pd.DataFrame, xaxis: str, title: str, scoring: list, xlabel='x', ylabel='y'):
    plt.figure(figsize=(10, 10)); plt.title(title, fontsize=13)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    ax = plt.gca()
    if df[xaxis].dtype == float:
        X_axis = np.array(df[xaxis], dtype=float)
    elif df[xaxis].dtype == object:
        X_axis = np.array(df[xaxis], dtype=str)
    for scorer, color in zip(sorted(scoring), ["g", "k"]):
        for sample, style in (("train", "--"), ("test", "-")):
            sample_score_mean = df["mean_%s_%s" % (sample, scorer)]
            sample_score_std = df["std_%s_%s" % (sample, scorer)]
            if sample == 'train':
                ax.fill_between( 
                    X_axis, sample_score_mean - sample_score_std, sample_score_mean + sample_score_std, alpha=0.1, color=color,
                )
            ax.plot(
                X_axis, sample_score_mean, style, color=color, alpha=1 if sample == "test" else 0.7, label="%s (%s)" % (scorer, sample),
            )
        best_index = np.nonzero(df["rank_test_%s" % scorer] == 1)[0][0]
        best_score = df["mean_test_%s" % scorer][best_index]
        ax.plot( [ X_axis[best_index], ] * 2, [0, best_score], linestyle="-.", color=color, marker="x", markeredgewidth=3, ms=8,)
        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))
    plt.legend(loc="best")
    plt.grid(False)
    plt.show()

def plot_categorical(df: pd.DataFrame, x: str, hue: str, ax: plt.Axes):
    sns.countplot(data=df, x=x, hue=hue, ax=ax)
    add_text = True if df[x].nunique() <= 5 else False
    xytext = lambda p: (0, 10) if p.get_height() < ax.get_yticks()[1] else (0, -10)
    if add_text:
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=xytext(p), textcoords='offset points')
    ax.set_xlabel("")
    text_kws = dict(transform=ax.transAxes, fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right')
    ax.text(x=0.97, y=0.96, s=f'Name: {x}', color='grey', **text_kws)


def plot_df(df: pd.DataFrame, cols=None, ncol=5, eta=4, hue=None, **kwargs):
    r""" The function plots distribution plots for each feature in the DataFrame.
    
    @param df:      The pandas DataFrame that contains the data to be plotted
    @param cols:    Specify which columns of the DataFrame should be plotted, optional
    @param ncol:    Specify the number of columns in the subplot grid, default 5
    @param eta:     Used to control the size of the figure. Increasing the value of eta will result in a larger figure, while decreasing it will result in a smaller figure, default 4

    ### Shape Measure of Distribution
    @measure skewness:      The asymmetry of a distribution. 0 suggests a symmetric distribution and positive value indicates a longer tail on the right side (right-skewed or mean > median)
    @measure kurtosis:      The peakedness compared to normal distribution (kurt=3), higher value indicates heavier tails and sharper peak
    """
    cols = list(df.columns if cols is None else cols)
    if hue:
        hue in cols and cols.remove(hue)
        cols = cols + [hue]
    df = df[cols]
    _, Ncol = df.shape
    Ncol = Ncol - (hue != None)
    a, b = divmod(Ncol, ncol)
    nrow = a + (b!=0)
    fig = plt.figure(**{**dict(figsize=(eta*nrow, eta*ncol)), **kwargs})   
    dtypes = df.dtypes
    for i in range(Ncol):
        ax = fig.add_subplot(nrow, ncol, i+1)
        if dtypes[i] == 'O':
            plot_categorical(df, x=cols[i], hue=hue, ax=ax)
        elif hue:
            sns.kdeplot(data=df, x=cols[i], hue=hue, fill=True, ax=ax)
        else:
            plot_df_helper(df.iloc[:, i], ax, cols[i])

def describe_df(df: pd.DataFrame):
    r"""Performs exploratory analysis on each variable in a DataFrame, providing information such as count, number of unique values, percentage of unique values, number of null values, and data type.
    """
    desc = pd.DataFrame(index=list(df))
    desc['count'] = df.count()
    desc['nunique'] = df.nunique()
    desc['%nunique'] = desc['nunique'] / len(df) * 100
    desc['null'] = df.isnull().sum()
    desc['type'] = df.dtypes
    return pd.concat([desc, df.describe().T.drop('count', axis=1)], axis=1)

def chi2_test(df: pd.DataFrame, category: str, target: str, alpha: float = 0.05):
    assert set([category, target]).issubset(set(df)), "make sure both category and target columns are in the dataset"
    chi2, p_value, _, _ = stats.chi2_contingency(
        df.groupby([category, target])
    )
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference in %s between %s classes."%(target, category))
    else:
        print("Fail to reject the null hypothesis. There is no significant difference in %s between %s classes."%(target, category))

def print_df(df: pd.DataFrame, limit=None, cols=None, title=None, caption=None, sort_col=None):
    r""" The function `print_df` is used to print a pandas DataFrame with rich API
    
    :param df:      pd.DataFrame to work on
    :param limit:   If `limit` is not specified or set to `None`, all rows in the DataFrame will be displayed
    :param cols:    If `cols` is not specified, all columns in the DataFrame will be included
    :param title:   Specify the title of the table
    :param caption: The caption for the table (underneath the table)
    :param sort_col:The column name to sort this table
    """
    def parse_str(text):
        text = str(text)
        text = re.sub(r'\[fg (.*)\](.*)\[/fg\]', r'[\1]\2[/]', text)
        text = re.sub(r'\[bg (.*)\](.*)\[/bg\]', r'[on \1]\2[/]', text)
        text = re.sub(r'(\d+,\s*\d+,\s*\d+)', lambda _: 'rgb(%s)'%re.sub(r"\s+", "", _.group(1)), text)
        return text

    df = df.reset_index().rename(columns={"index": ""})
    if sort_col is not None:
        df = df.sort_values(by=sort_col, ascending=False)
    if cols is None: cols = list(df)
    df = df[cols]
    tbl = Table(show_footer=False, title=title)
    #> step 1: add columns
    for col in df.columns: 
        tbl.add_column(col)
    #> step 2: add rows
    for row in df.values[:limit]: 
        tbl.add_row(*list(map(parse_str, row)))
    #> step 3: 
    #>  a) move text to right
    #>  b) add random color (header + bold)
    for i,col in enumerate(tbl.columns): 
        col.justify = 'right'
        col.header_style = ('bold ' + COLORS[i % NCOLOR])
        col.style = COLORS[i % NCOLOR]
    #> step 4: change the table border style
    tbl.border_style = 'bright_black'
    #> step 5: add captions
    if caption is not None: tbl.caption = caption
    rprint(tbl)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn.utils import shuffle
    boston = datasets.load_breast_cancer()
    X,y = MC(n_samples=10_000, n_features=20, random_state=42, n_informative=5, weights=[0.9]) # 10% positive
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    bench = Benchmark()
    result = bench.fit(X_train, y_train, X_test, y_test)
    print(1)
    


