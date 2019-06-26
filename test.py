import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from simple.distributions import *


### central limit theorem
samples = []
for i in range(1000):
    s = 0
    for _ in range(5000):
        s += flip(0.5)

    samples.append(s / 5000.0)

"""
plt.hist(samples, bins=50, density=True)
plt.show()
"""

### gamma distribution
samples_np = []
samples_test = []
for i in range(5000):
    x = np.random.gamma(shape=3, scale=1.0)
    y = gamma(alpha=3, beta=1)
    samples_np.append(x)
    samples_test.append(y)

plt.hist(samples_np, bins=50, density=True, label="np.random.gamma")
plt.hist(samples_test, bins=50, density=True, label="distributions.gamma")
plt.legend(loc="best")
plt.show()