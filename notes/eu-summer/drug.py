import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm



### DATA
drug = pd.DataFrame(dict(iq=(101,100,102,104,102,97,105,105,98,101,100,123,105,103,100,95,102,106,
        109,102,82,102,100,102,102,101,102,102,103,103,97,97,103,101,97,104,
        96,103,124,101,101,100,101,101,104,100,101),
                         group='drug'))
placebo = pd.DataFrame(dict(iq=(99,101,100,101,102,100,97,101,104,101,102,102,100,105,88,101,100,
           104,100,100,100,101,102,103,97,101,101,100,101,99,101,100,100,
           101,100,99,101,100,102,99,100,99),
                            group='placebo'))
trial_data = pd.concat([drug, placebo], ignore_index=True)
trial_data.hist('iq', by='group')
plt.show()

### MODEL
with pm.Model() as model:
    mu_0    = pm.Normal("mu_0", 100, sd=10)
    mu_1    = pm.Normal("mu_1", 100, sd=10)
    sigma_0 = pm.Uniform("sigma_0", lower=0, upper=20)
    sigma_1 = pm.Uniform("sigma_1", lower=0, upper=20)
    v       = pm.Exponential("v_minus_one", 1/29) + 1
    # drug distributions
    drug_like    = pm.StudentT("drug_like", nu=v, mu=mu_1, lam=sigma_1**-2, observed=drug.iq)
    placebo_like = pm.StudentT("placebo_like", nu=v, mu=mu_0, lam=sigma_0**-2, observed=placebo.iq)
    # measurement of effects
    diff_of_means = pm.Deterministic("difference of means", mu_1-mu_0)
    effect_size   = pm.Deterministic("effect size", diff_of_means / np.sqrt((sigma_1**2 + sigma_0**2) / 2))

### INFERENCE
with model:
    trace = pm.sample(1000)

pm.plot_posterior(trace[100:],
    varnames=["difference of means", "effect size"],
    rev_val=0,
    color="#87ceeb")
plt.show()