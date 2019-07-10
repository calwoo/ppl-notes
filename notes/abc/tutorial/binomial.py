import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-pastel")


### DATA
num_trials = 1000
p = 0.7

### MODEL
def flip(p=0.5):
    """
    Bernoulli distribution, x ~ Bernoulli(p)
    """
    return int(uniform(0,1) < p)

def uniform(low=0, high=1):
    """
    x ~ uniform(low, high)
    """
    length = high - low
    return length * np.random.rand() + low

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

### INFERENCE
samples = []
num_samples = 10000

while len(samples) < num_samples:
    # sample p from prior
    param = beta(1,1)
    # simulate data from this parameter
    simulation = binomial(num_trials, param)
    # compare to expected data
    if simulation == int(p * num_trials):
        samples.append(param)
        

### PLOT
plt.hist(samples, bins=100, density=True)
plt.xlim(0,1)
plt.show()
