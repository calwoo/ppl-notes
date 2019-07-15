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

### BERNOULLI
class Dist:
    # Discrete distribution class.
    def sample(self):
        pass

    def support(self):
        pass

class Bernoulli(Dist):
    def __init__(self, p=0.5):
        self.p = p
    
    def sample(self):
        return int(uniform(0,1) > 0.5)

    def support(self):
        return [0, 1]

    def score(self, s):
        return np.log(0.5)