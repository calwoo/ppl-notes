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

def exponential(lamb=1):
    """
    Sampling from the exponential distribution using
    inverse transform sampling.
        x ~ Exp(lambda)
    """
    u = uniform(0,1)
    x = -(1/lamb) * np.log(1-u)
    return x

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

def beta(a=1, b=1):
    """
    Beta distribution implementation from R. C. H. Cheng's
    “Generating Beta Variates with Nonintegral Shape Parameters”
    (1978).  x ~ Beta(a,b)"""
    # set alpha, beta, gamma
    alpha = a + b
    if min(a,b) <= 1:
        bet = max(1/a, 1/b)
    else:
        bet = np.sqrt((alpha-2)/(2*a*b-alpha))
    gamma = a + 1/bet
    # loop until
    while True:
        u1 = uniform(0,1)
        u2 = uniform(0,1)
        v = bet * np.log(u1/(1-u1))
        w = a * np.exp(v)
        if alpha*np.log(alpha/(b+w)) + gamma*v - 1.3862944 >= np.log(u1**2*u2):
            x = w / (b+w)
            return x
        


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
    if flip(p):
        return 1
    else:
        return 1 + geometric(p)

def binomial(n, p=0.5):
    """
    Iterative binomial distribution, models
    the number of successes in n trials with
    probability of success p,
        k ~ Bin(n,p) 
    """
    successes = 0
    for _ in range(n):
        if flip(p):
            successes += 1
    return successes