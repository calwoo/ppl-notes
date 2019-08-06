import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import theano.tensor as tt

challenger_data = np.array([
    [ 66.  , 0.],
    [ 70.  , 1.],
    [ 69.  , 0.],
    [ 68.  , 0.],
    [ 67.  , 0.],
    [ 72.  , 0.],
    [ 73.  , 0.],
    [ 70.  , 0.],
    [ 57.  , 1.],
    [ 63.  , 1.],
    [ 70.  , 1.],
    [ 78.  , 0.],
    [ 67.  , 0.],
    [ 53.  , 1.],
    [ 67.  , 0.],
    [ 75.  , 0.],
    [ 70.  , 0.],
    [ 81.  , 0.],
    [ 76.  , 0.],
    [ 79.  , 0.],
    [ 75.  , 1.],
    [ 76.  , 0.],
    [ 58.  , 1.]
])

plt.scatter(challenger_data[:, 0], challenger_data[:, 1], s=75, color="k",
            alpha=0.5)
plt.yticks([0, 1])
plt.ylabel("Damage Incident?")
plt.xlabel("Outside temperature (Fahrenheit)")
plt.title("Defects of the Space Shuttle O-Rings vs temperature")
plt.show()

# model
temperature = challenger_data[:, 0]
defect = challenger_data[:, 1]

with pm.Model() as model:
    beta = pm.Normal("beta", mu=0, tau=0.001, testval=0)
    alpha = pm.Normal("alpha", mu=0, tau=0.001, testval=0)
    p = pm.Deterministic("p", 1.0/(1.0 + tt.exp(beta*temperature + alpha)))

    observed = pm.Bernoulli("bernoulli_obs", p, observed=defect)
    # MAP estimate
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(120000, step=step, start=start)
    burned_trace = trace[100000::2]

alpha_samples = burned_trace["alpha"][:, None]  # best to make them 1d
beta_samples = burned_trace["beta"][:, None]

#histogram of the samples:
plt.subplot(211)
plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\beta$", color="#7A68A6", normed=True)
plt.legend()

plt.subplot(212)
plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\alpha$", color="#A60628", normed=True)
plt.legend()
plt.show()