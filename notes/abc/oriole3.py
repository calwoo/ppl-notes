import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import itertools
import random


def abayes(data, prior_sampler, simulate, compare):
    """
    generator that yields samples from the posterior by
    approximate bayesian computation.
    """
    for p in prior_sampler:
        if compare(data, simulate(p)):
            yield p

# take function from functional programming
def take(n, iterable):
    """get the first n items of an iterable as a list
    """
    return list(itertools.islice(iterable, n))

# compare
def compare_conversion_fuzzy(obs, sim_data, tol=0.1):
    """
    compares observations, but now up to a tolerance
    """
    return abs(sim_data-obs)/(sim_data + 1.0) < tol

# simulate
def simulate_conversion(p, n_visitors):
    """returns number of visitors who convert given conversion
    fraction p.
    """
    outcomes = (np.random.rand() < p for _ in range(n_visitors))
    return sum(outcomes)
    
# prior sampler
def uniform_prior_sampler():
    """generator for p ~ uniform(0,1).
    This is an infinite generator.
    """
    while True:
        yield np.random.random()

# posterior sampler
posterior_sampler = abayes(
    data=40,
    prior_sampler=uniform_prior_sampler(),
    simulate=lambda p: simulate_conversion(p, 1000),
    compare=compare_conversion_fuzzy)

### german tank problem
captured_tanks = [314, 421]

def prior_ntanks_sampler(captured_tanks, upper=5000):
    """generator for random integers in the range
    (max(captured_tanks), upper)
    """
    while True:
        yield np.random.randint(max(captured_tanks), upper)

def simulate_tanks(n_tanks, n_caught):
    """return serial numbers of n_caught tanks given
    are n_tanks
    """
    return random.sample(range(n_tanks), n_caught)

def compare_tanks(obs, sim, tol=20):
    """
    compares observations, but up to a tolerance
    """
    return all(abs(o-s) <= tol for o, s in zip(sorted(obs), sorted(sim)))

# posterior
posterior_ntanks_sampler = abayes(
    data=captured_tanks,
    prior_sampler=prior_ntanks_sampler(captured_tanks),
    simulate=lambda n: simulate_tanks(n, len(captured_tanks)),
    compare=compare_tanks)

# plot
tank_samples = take(1000, posterior_ntanks_sampler)

tank_bins = range(0, 5000, 50)
plt.hist(tank_samples, density=True, bins=tank_bins)
plt.show()

### pymc3
with pm.Model():
    n_tanks = pm.DiscreteUniform("num_tanks", lower=max(captured_tanks), upper=5000)
    obs = pm.DiscreteUniform("obs", lower=0, upper=n_tanks, observed=captured_tanks)
    trace = pm.sample(10000)

burn_in = 1000
plt.hist(trace[burn_in:].get_values('num_tanks'), density=True, bins=100)
plt.show()