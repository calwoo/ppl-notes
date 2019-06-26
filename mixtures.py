"""
A gaussian-gamma mixture model from Wingate et al.
"""

import numpy as np
import matplotlib.pyplot as plt

from simple.distributions import *


def ggmixture(num_samples=1000):
    samples = []
    for _ in range(num_samples):
        if flip(p=0.5):
            x = normal(0,1)
        else:
            x = gamma(1,1)
        samples.append(x)
    return samples


samples = ggmixture(5000)
plt.hist(samples, bins=100, density=True)
plt.show()