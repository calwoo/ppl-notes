import pyro.distributions as dist
from pyro import sample
import torch

"""
A probabilistic program is a problem that samples from a distribution.
"""
# Fair coin
flip = sample("coinflip", dist.Bernoulli(probs=0.5))

# Noise
noise = sample("noise", dist.Normal(loc=0, scale=1))

"""
In Pyro, elementary random primitives (distributions) have two main
functions: sample and log_prob.
"""
test = dist.Bernoulli(probs=0.5).log_prob(torch.tensor(0.0)).exp()

"""
We can describe probabilistic programs as generative models.
"""
def sleep_model():
    """
    Models how many hours one sleeps in a day.
    """
    lazy = sample("lazy", dist.Bernoulli(0.3))
    if lazy:
        # If lazy, will ignore alarm with some probability
        ignore_alarm = sample("ignore_alarm", dist.Bernoulli(0.8))
        sleep_length = sample("sleep_length", dist.Normal(8 + 2*ignore_alarm, 1))
    else:
        sleep_length = sample("sleep_length", dist.Normal(6, 1))
    return sleep_length

# To test above, use interactive mode `python -i generative.py`.