"""
1. What are the key components of a power analysis, and how do they interact to determine the required sample size for a study?

A: minimum detectable effect size, significance level, desired power, standard deviation and sample size

2. Describe the trade-off between Type I and Type II errors in hypothesis testing and how it relates to power analysis. 

Type I error (alpha) is the probability of rejecting a null hypothesis when it is actually true. In other words, it's a false positive. Lowering α makes it harder to declare an effect statistically significant, reducing the risk of false positives but increasing the risk of false negatives.

Type II Error (beta): This is the probability of failing to reject a null hypothesis when it is actually false. It's a false negative. Power (1 - beta) is the complementary probability, and it represents the likelihood of correctly detecting an effect when it truly exists. 

![](https://www.theanalysisfactor.com/wp-content/uploads/2009/12/one-sided-test.jpg)
"""
__all__ = ['power_analysis']

import scipy.stats as stats
import numpy as np


def power_analysis(mde, alpha, power, sigma, two_sided=True, kappa=1):
    """ The `power_analysis` function calculates the sample size needed for a power analysis based on the minimum detectable effect size, significance level, desired power, standard deviation, and other parameters.
    
    #### Parameter
    @param mde      Minimum Detectable Effect (difference between mu)
    @param alpha    The type i false positive error rate
    @param power    Statistical power is the probability of correctly rejecting the null hypothesis when it is false (1-power = beta = typeii error or false negative)
    @param sigma    The standard deviation of the population.
    @param two_sided    The parameter "two_sided" is a boolean value that determines whether the hypothesis test is two-sided or one-sided.
    
    @return three values: the total sample size (nA + nB), the sample size for group A (nA), and the sample size for group B (nB).
    """
    effect_size = mde # absolute effect (mu_alternative-mu_h0)
    ceil = lambda _: int(np.ceil(_))
    if two_sided:
        alpha_test = alpha / 2
    else:
        alpha_test = alpha
    z_alpha = stats.norm.ppf(1-alpha_test)
    z_power = stats.norm.ppf(power)
    a, b = (z_power+z_alpha)**2, (sigma/effect_size)**2
    nA, nB = (1+kappa)*a*b, (1+1/kappa)*a*b
    return ceil(nA)+ceil(nB), ceil(nA), ceil(nB)
