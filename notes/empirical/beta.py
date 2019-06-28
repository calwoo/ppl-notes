import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import gamma
plt.style.use("seaborn-pastel")

# a simple beta function
def beta(x, a, b):
    denom = gamma(a) * gamma(b) / gamma(a + b)
    numer = x**(a-1) * (1-x)**(b-1)
    return numer / denom

# some beta distributions
x = np.linspace(0,1,1000)

plt.plot(x, beta(x, a=1, b=2), label="a=1, b=2")
plt.plot(x, beta(x, a=3, b=3), label="a=3, b=3")
plt.plot(x, beta(x, a=20, b=20), label="a=20, b=20")
plt.plot(x, beta(x, a=50, b=10), label="a=50, b=10")
plt.legend(loc="best")
plt.show()

"""
What is the beta distribution good for? Notice it has support on
(0,1). So it can be thought of as a "probability distribution of
probabilities.
"""

# here Beta(81, 219) describes the probability of success in a bernoulli trial
# with so far 81 successes and 219 failures
plt.plot(x, stats.beta.pdf(x, a=81, b=219), label="a=81, b=219")
plt.show()

# if we observe via a bernoulli trial 100 successes out of 300 trials, our
# beta posterior updates as Beta(81 + 100, 219 + 200)
plt.plot(x, stats.beta.pdf(x, a=81, b=219), label="a=81, b=219")
plt.plot(x, stats.beta.pdf(x, a=181, b=419), label="a=181, b=419")
plt.legend(loc="best")
plt.show()

"""
This is because the beta distribution is the conjugate prior of the
Bernoulli distribution.

We can also see this by simulation.
"""

