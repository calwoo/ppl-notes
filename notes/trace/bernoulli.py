import torch
import random

# Implementation of a simple Bernoulli distribution with API
# like in Pyro.

class Bernoulli:
    def __init__(self, p):
        self.p = p

    def sample(self):
        if random.random() < self.p:
            return torch.tensor(1.0)
        else:
            return torch.tensor(0.0)

    def log_prob(self, x):
        return torch.log(x * self.p + (1 - x) * (1 - self.p))

b = Bernoulli(0.8)