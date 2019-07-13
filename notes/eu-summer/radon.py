import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm


### DATA
data  = pd.read_csv("data/radon.csv", index_col=0)
query = data.query('county=="HENNEPIN"').log_radon

### MODEL
with pm.Model() as model:
    mu    = pm.Normal("mu", mu=0, sd=10)
    sigma = pm.Uniform("sigma", 0, 10)
    y     = pm.Normal("y", mu=mu, sigma=sigma, observed=query)

### INFERENCE
with model:
    samples = pm.fit().sample(1000)

pm.plot_posterior(samples, varnames=["mu"], ref_val=np.log(4), color='LightSeaGreen')
plt.show()