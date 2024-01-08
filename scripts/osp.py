"""
Early Optimal Stopping Problem
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import time
import functools
from typing import List
from datetime import datetime
from pathos.multiprocessing import ProcessingPool as Pool

@functools.lru_cache(maxsize=None)  # Set maxsize to None for an unbounded cache
def shepp_urn(p=20, n=10):
    """
    the probability of the next realization is known ahead
    @early: number of object is finite and if all of them are drawn, payoff is not optimal
    @optimal: more sample we draw, less unknown 
    """
    if p == 0: return 0
    if n == 0: return p
    prob = p/(p+n)
    return prob * (1+shepp_urn(p-1, n)) + (1-prob) * (-1+shepp_urn(p, n-1))

def shepp_entropy_helper(p, n):
    if p + n == 0: return 0
    entropy, prob = 0, p/(p+n)
    entropy -= 0 if p == 0 else prob*np.log2(prob)
    entropy -= 0 if n == 0 else (1-prob)*np.log2(1-prob)
    return entropy

def shepp_entropy(p_start, n_start, n_realization):
    games = shepp_realization(p_start, n_start, n_realization)
    num_pos, num_neg = p_start-np.cumsum(games==1, axis=1), n_start-np.cumsum(games==-1, axis=1)
    entropy_np = np.vectorize(shepp_entropy_helper)(num_pos, num_neg)
    plt.figure(figsize=(6, 6))  # Adjust the figure size as needed
    df = wide_to_long(entropy_np, 'entropy')
    g = sns.lineplot(data=df, x='step', y='value')
    plt.savefig(os.path.normpath(os.path.join(__file__, '..', 'shepp_entropy.png')))


def shepp_realization(p_start, n_start, n_realization=50):
    realizations = np.zeros((n_realization, p_start + n_start))
    ones = np.ones(p_start)
    neg_ones = -1 * np.ones(n_start)
    realization = np.concatenate((ones, neg_ones))
    for i in range(n_realization):
        np.random.shuffle(realization)
        realizations[i] = np.copy(realization)
    return realizations

def shepp_payoff(p_start, n_start, n_realization=50, do_plot=True):
    games = shepp_realization(p_start, n_start, n_realization)
    payoff_ne = np.cumsum(games, axis=1)
    num_pos, num_neg = p_start-np.cumsum(games==1, axis=1), n_start-np.cumsum(games==-1, axis=1)
    def early_stopping(xp, xn, arr):
        index = len(arr)-1  # stop here
        for i, x in enumerate(zip(xp, xn)): 
            if shepp_urn(*x) < 0: index = i; break
        arr = np.copy(arr)
        arr[index+1:] = arr[index]
        return arr
    payoff_es = np.vectorize(early_stopping, signature='(n),(n),(n)->(n)')(num_pos, num_neg, payoff_ne)
    if do_plot:
        non_stop = wide_to_long(payoff_ne, 'no')
        early_stop = wide_to_long(payoff_es, 'yes')
        df = pd.concat((non_stop, early_stop), axis=0)
        plt.figure(figsize=(6, 6))  # Adjust the figure size as needed
        g = sns.lineplot(data=df, x='step', y='value', hue='stop')
        plt.savefig(os.path.normpath(os.path.join(__file__, '..', 'shepp_payoff.png')))

    
def wide_to_long(array, name, type_name='stop'):
    # assert array.shape = (a, b, c)
    df = pd.DataFrame(array.T)
    df = df.stack().reset_index()
    df.columns = ['step', 'id', 'value']
    df[type_name] = name
    return df


class Simulation:
    """Look-then-leap simulations.
    """

    def __init__(self,
                 candidates: List[int],
                 threshold: float = 0.3):
        """
        Args:
            candidates: List of incoming candidates, represented by their
                (hidden) rankings. The higher the better.
            threshold (float): Proportion of `look` period. 0.3 means one will
                look 30% of the total candidates before leap.
        """

        self.candidates = candidates
        assert(0. < threshold < 1.)
        if not len(self.candidates) == len(set(self.candidates)):
            raise ValueError("Candidate should have unique rankings.")

        self.num_candidates = len(self.candidates)
        self.num_looks = int(threshold * self.num_candidates) + 1
        self.true_best = np.min(self.candidates)
        self.selected = None

    def run(self):
        """Simulate the interview process.
        """
        self.best_so_far = np.min(self.candidates[:self.num_looks])

        for i in self.candidates[self.num_looks:]:
            self.selected = i
            if i < self.best_so_far:
                break  # we are done!

    def evaluate(self):
        """Returns True if we do get the best candidate.
        """
        if self.selected is None:  # be careful of 0!
            raise ValueError("Please run the simulation first.")

        if self.selected == self.true_best:
            return True
        elif self.selected > self.true_best:
            return False
        else:
            raise RuntimeError("WTF?")


class Trials:
    """Runs multiple trails of the simulation.
    """
    def __init__(self,
                 n_trials: int = 100,
                 **kwargs):
        """Setting a single trial.
        """
        self.n_trials = n_trials
        self.n_jobs = kwargs.get('n_jobs', 10)
        self.threshold = kwargs.get('threshold', 100)
        self.n_candidates = kwargs.get('n_candidates', 100)
        self.results = []

    def single_run(self, x):
        """Single simulation, with n_candidates.

        The argument x is to satisfy later multiprocess runs.
        """
        np.random.seed(datetime.now().microsecond)
        candidates = np.random.permutation(self.n_candidates)

        S = Simulation(candidates, self.threshold)
        S.run()

        return S.evaluate()

    def run(self):
        """Runs all trials in mp fashion.
        """
        self.results = Pool(self.n_jobs).map(
            self.single_run,
            range(self.n_trials))

def select_candidate():
    """
        threshold:  the first N x threshold candidates will be screened to help decision making
        success:    the chance we will select the best candidate after screening
    """
    n_trials = 10000  # number of trials at each threshold
    n_candidates = 1000  # number of candidate for each simulation
    thresholds = np.arange(0.01, 1., 0.01)
    results_all_thresholds = []
    for i, t in enumerate(thresholds):
        print('Testing threshold {} out of {}...'.format(i + 1, len(thresholds)), end='\r')
        T = Trials(n_trials=n_trials, n_candidates=n_candidates, threshold=t, n_jobs=12)
        T.run()
        results_all_thresholds.append(T.results)
    rate_mean = np.mean(results_all_thresholds, axis=1)
    rate_std = [x * (1 - x) / np.sqrt(n_trials) for x in rate_mean]
    plt.figure(figsize=(8, 4))  # Adjust the figure size as needed
    g = sns.scatterplot(x=thresholds, y=rate_mean, color='royalblue', alpha=0.7)
    x = np.arange(0.01, 1., 0.001)
    y = [-1. * e * np.log(e) for e in x]
    g1 = sns.lineplot(x=x, y=y, color='firebrick', alpha=0.7)
    g2 = sns.lineplot(x=[np.exp(-1)] * 2, y=[0, 0.4], color='blue', alpha=0.9, linewidth=2.5)
    plt.xlabel('Threshold'); plt.ylabel('Success rate'); plt.tight_layout()
    plt.savefig('/home/dalab1/project/blog/mystuff/_src/Latex/papers/images/secretary_problem.png')



if __name__ == '__main__':
    p, n = 10, 10
    start_time = time.time()
    # shepp_payoff(p, n, n_realization=100)
    # shepp_entropy(p, n, n_realization=100)
    select_candidate()
    print(f"it costs {(time.time() - start_time)/1000} seconds")
    