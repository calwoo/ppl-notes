import pyro.distributions as dist
from pyro.poutine import trace
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro import sample, condition
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

from generative import *

"""
To perform variational inference we need a parameterized family of distributions q(z)
over the latent variables of our generative model to represent the approximate
posterior. In Pyro, these functions are called "guides".
"""
def bad_sleep_guide():
    # Dirac delta guide
    sample("lazy", dist.Delta(1.0))
    sample("ignore_alarm", dist.Delta(0.0))

def sleep_guide():
    # Constraint on probability
    valid_prob = pyro.constraints.interval(0.0, 1.0)
    # Bernoulli probabilities
    lazy_p = pyro.param("lazy_p", torch.tensor(0.8), constraint=valid_prob)
    ia_p   = pyro.param("ia_p", torch.tensor(0.9), constraint=valid_prob)
    # Guide model
    lazy = sample("lazy", dist.Bernoulli(lazy_p))
    if lazy == 1:
        sample("ignore_alarm", dist.Bernoulli(ia_p))

sleep_guide()

"""
In general, to do stochastic variational inference in Pyro it's fairly easy.
"""
# Conditioned model on observed data
underslept = condition(sleep_model, {"amount_slept": 6})

optimizer = Adam({"lr": 0.005, "betas": (0.9, 0.999)})
svi = SVI(underslept, sleep_guide, optimizer, loss=Trace_ELBO())

param_vals = []
for _ in range(1000):
    svi.step()
    param_vals.append({k: param(k).item() for k in ["lazy", "ignore_alarm"]})