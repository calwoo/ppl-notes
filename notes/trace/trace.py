import pyro.distributions as dist
from pyro.poutine import trace
from pyro import sample, condition
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

from generative import *

"""
To answer queries about probabilistic programs (inference), we need
to collect traces.
"""
tr = trace(sleep_model).get_trace()

# What does this trace look like?
for name, properties in tr.nodes.items():
    if properties["type"] == "sample":
        pprint({
            name: {
                "value": properties["value"],
                "prob" : properties["fn"].log_prob(properties["value"]).exp()}
        })

exec_prob = tr.log_prob_sum().exp()

# To compute the trace of an observation, we use pyro.condition
conditioned_model = condition(sleep_model, {
    "lazy"        : torch.tensor(1.0),
    "ignore_alarm": torch.tensor(0.0),
    "amount_slept": torch.tensor(8.2)
})

conditional_prob = trace(conditioned_model).get_trace().log_prob_sum().exp()

"""
Manipulating traces allows us to solve any sort of inference query just
by repeated sampling. For example, to compute a marginal distribution over
each of the variables, we merely "sum out" all other variables in a large
collection of sampled traces.
"""
traces = []
for _ in range(1000):
    # Run the model, collecting all sampled variables in a trace
    tr = trace(sleep_model).get_trace()
    # Retrieve the values of each of the relevant vars
    values = {}
    for name, ptys in tr.nodes.items():
        if ptys["type"] == "sample":
            values[name] = ptys["value"].item()
    traces.append(values)

# Plot histogram of values (which corresponds to each marginal distribution)
pd.DataFrame(traces).hist()
plt.show()