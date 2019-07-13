import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm


### DATA
disasters_data = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                            2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                            3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

num_years = len(disasters_data)

plt.figure(figsize=(12.5, 3.5))
plt.bar(np.arange(1851, 1962), disasters_data, color="#348ABD")
plt.xlabel("Year")
plt.ylabel("Disasters")
plt.title("UK coal mining disasters, 1851-1962")
plt.xlim(1851, 1962)
plt.show()

### MODEL
with pm.Model() as model:
    switchpoint = pm.DiscreteUniform("switchpoint", lower=0, upper=num_years)
    early_mean  = pm.Exponential("early_mean", 1)
    late_mean   = pm.Exponential("late_mean", 1)
    rate        = pm.math.switch(switchpoint <= np.arange(num_years), early_mean, late_mean)
    disasters   = pm.Poisson("disasters", mu=rate, observed=disasters_data)

### INFERENCE
with model:
    trace = pm.sample(1000, tune=1000, init=None)

pm.traceplot(trace, varnames=["early_mean", "late_mean", "switchpoint"])
plt.show()