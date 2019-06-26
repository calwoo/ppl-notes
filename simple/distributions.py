"""
A collection of simple distributions, built from ERPs (elementary
random primatives) from the numpy.random library.
"""

import numpy as np

### CONTINUOUS
def uniform(low=0, high=1):
    """
    x ~ uniform(low, high)
    """
    length = high - low
    return length * np.random.rand() + low

def normal(mean=0, std=1):
    """
    Univariate gaussian, x ~ N(mu, sigma^2)
    """
    return std * np.random.randn() + mean

def gamma(alpha, beta=1):
    """
    Gamma distribution implementation from Marsaglia-Tsang,
    "A simple method for generating gamma variables".
        x ~ Gamma(alpha, beta)
    """
    # set d, c
    d = alpha - 1/3
    c = 1 / np.sqrt(9*d)
    # loop until
    while True:
        # draw ERPs independently
        z = normal(0,1)
        u = uniform(0,1)
        # condition
        v = (1 + c*z)**3
        if z > -1/c and np.log(u) < 0.5*z**2 + d - d*v + d*np.log(v):
            ans = d*v
            return ans / beta


### DISCRETE
def flip(p=0.5):
    """
    Bernoulli distribution, x ~ Bernoulli(p)
    """
    return int(uniform(0,1) > 0.5)

def geometric(p=0.5):
    """
    Recursive geometric distribution, x ~ Geometric(p)
    """
    pass