import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

with pm.Model() as model:
    p = pm.Uniform("p", lower=0, upper=1)

p_true = 0.05
N = 1500
occurrences = stats.bernoulli.rvs(p_true, size=N)

print(occurrences)

with model:
    obs = pm.Bernoulli("obs", p, observed=occurrences)
    # trace
    step = pm.Metropolis()
    trace = pm.sample(18000, step=step)
    burned_trace = trace[1000:]

plt.figsize(12.5, 4)
plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
plt.hist(burned_trace["p"], bins=25, histtype="stepfilled", normed=True)
plt.legend()
plt.show()