import numpy as np
import matplotlib.pyplot as plt
import itertools
import random


# python ABC
def abayes(data, prior_sampler, simulate, compare):
    """
    generator that yields samples from the posterior by
    approximate bayesian computation.
    """
    for p in prior_sampler:
        if compare(data, simulate(p)):
            yield p


# redo a/b test
n_visitors_a = 100
n_conv_a = 4

n_visitors_b = 40
n_conv_b = 2

def compare_conversion(obs, sim_data):
    """
    compares observations. simple!
    """
    return obs == sim_data

def simulate_conversion(p, n_visitors):
    """returns number of visitors who convert given conversion
    fraction p.
    """
    outcomes = (np.random.rand() < p for _ in range(n_visitors))
    return sum(outcomes)
    
def uniform_prior_sampler():
    """generator for p ~ uniform(0,1).
    This is an infinite generator.
    """
    while True:
        yield np.random.random()


# take function from functional programming
def take(n, iterable):
    """get the first n items of an iterable as a list
    """
    return list(itertools.islice(iterable, n))


# running abayes
posterior_a_sampler = abayes(
    data=n_conv_a,
    prior_sampler=uniform_prior_sampler(),
    simulate=lambda p: simulate_conversion(p, n_visitors_a),
    compare=compare_conversion)


# layout b test -- normal distribution prior
def normal_prior_sampler(mu=0.06, sigma=0.02):
    """generator for stream of samples
            p ~ Normal(mu, sigma)
    """
    while True:
        p = sigma * np.random.randn() + mu
        if 0 <= p <= 1:
            yield p

posterior_b_sampler = abayes(
    data=n_conv_b,
    prior_sampler=normal_prior_sampler(),
    simulate=lambda p: simulate_conversion(p, n_visitors_b),
    compare=compare_conversion)

num_samples = 5000
a_samples = take(num_samples, posterior_a_sampler)
b_samples = take(num_samples, posterior_b_sampler)

# plot posteriors
abbins = [i/200.0 for i in range(50)]  # 50 bins between 0 and 0.25
plt.hist(a_samples, bins=abbins, label='A', density=True)
plt.hist(b_samples, bins=abbins, label='B', alpha=0.5, density=True)
plt.title('Estimates of conversion fraction after the A/B test')
plt.legend()
plt.show()

"""
The above setup allows us to do some online learning.
"""

conversions = [1, 0, 2, 0, 1]

def online_abayes(datas, prior_sampler, simulate, compare, num_samples=10000):
    """generator that yields samples from the posterior
    for each online observation in the data
    """
    for data in datas:
        posterior_samples = take(num_samples, abayes(
            data,
            prior_sampler,
            simulate,
            compare))
        yield posterior_samples
        prior_samples = posterior_samples
        prior_sampler = sampler_from_samples(prior_samples)

def sampler_from_samples(samples):
    """
    From a list of samples, returns a generator that
    repeatedly resamples from the list.
    """
    samples = list(samples)
    random.sample(samples, len(samples))
    return itertools.cycle(samples)


# create generator from online observations
posteriors = online_abayes(
    datas=conversions,
    prior_sampler=uniform_prior_sampler(),
    simulate=lambda p: simulate_conversion(p, 20),
    compare=compare_conversion)

# plot
fig, ax = plt.subplots(1, len(conversions), sharey=True)
for i, p in enumerate(posteriors):
    ax[i].hist(p, bins=abbins, density=True)
    ax[i].set_title('{} visitors'.format((i+1)*20))
fig.tight_layout()
plt.show()