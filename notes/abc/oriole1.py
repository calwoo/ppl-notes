import numpy as np
import matplotlib.pyplot as plt


# A/B test data
n_visitors_a = 100
n_conv_a = 4

n_visitors_b = 40
n_conv_b = 2

# approximate bayesian computation
def estimate_conversion(n_visitors, n_conv, trial_conversion, n_estimates=5000):
    """
    performs ABC-- samples n_estimates estimates of the conversion fraction
    assuming we are in a situation where we have n_visitors, n_conv number of
    converts.
    """
    i = 0
    estimates = []
    while i < n_estimates:
        # 1. generate a trial value for parameter we want to compute
        p = trial_conversion()
        # 2. simulate the data assuming trial value
        n_sim = simulate_conversion(p, n_visitors)
        # 3. if simulation looks like real data, keep trial value
        if n_conv == n_sim:
            estimates.append(p)
            i += 1
    return estimates

def trial_conversion_a():
    """what if we know nothing about the conversion fraction?
    """
    return np.random.rand()

def simulate_conversion(p, n_visitors):
    """returns number of visitors who convert given conversion
    fraction p.
    """
    outcomes = [np.random.rand() < p for _ in range(n_visitors)]
    return sum(outcomes)


# run test
a_estimates = estimate_conversion(n_visitors_a, n_conv_a, trial_conversion_a)
abbins = [i/200.0 for i in range(50)]  # 50 bins between 0 and 0.25

plt.hist(a_estimates, bins=abbins, density=True)
plt.title('Estimates of conversion fraction for A after the A/B test')
plt.show()

# prior for b
def trial_conversion_b():
    while True:
        x = 0.02 * np.random.randn() + 0.06
        if 0 <= x <= 1:
            return x

trial_as = [trial_conversion_a() for _ in range(100000)]
trial_bs = [trial_conversion_b() for _ in range(100000)]

plt.hist(trial_as, bins=abbins, label='A', normed=True)
plt.hist(trial_bs, bins=abbins, label='B', alpha=0.5, normed=True)
plt.title('Beliefs about conversion fraction prior to A/B test')
plt.legend()
plt.show()

# abc with this prior for b
b_estimates = estimate_conversion(n_visitors_b, n_conv_b, trial_conversion_b)
plt.hist(a_estimates, bins=abbins, label='A', normed=True)
plt.hist(b_estimates, bins=abbins, label='B', alpha=0.5, normed=True)
plt.title('Estimates of conversion fraction after the A/B test')
plt.legend()
plt.show()