import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# data
size = 200
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
true_reg_line = true_intercept + true_slope * x
y = true_reg_line + np.random.normal(scale=0.5, size=size)

plt.figure(figsize=(7,7))
plt.scatter(x, y, marker="x", label="data", c="g")
plt.plot(x, true_reg_line, label="true regression", lw=2.0, c="m")
plt.legend(loc="best")
plt.show()

# model
with pm.Model() as model:
    # priors
    sigma = pm.HalfCauchy("sigma", beta=10, testval=1)
    intercept = pm.Normal("Intercept", mu=0, sigma=20)
    slope = pm.Normal("x", mu=0, sigma=20)
    # likelihood
    likelihood = pm.Normal("y", mu=intercept + slope * x,
                            sigma=sigma, observed=y)

    # inference
    trace = pm.sample()

plt.figure(figsize=(7,7))
pm.traceplot(trace)
plt.tight_layout()

plt.figure(figsize=(7, 7))
plt.plot(x, y, 'x', label='data')
pm.plots.plot_posterior_predictive_glm(trace, samples=100, 
                                    label='posterior predictive regression lines')
plt.plot(x, true_reg_line, label='true regression line', lw=3., c='y')

plt.title('Posterior predictive regression lines')
plt.legend(loc=0)
plt.xlabel('x')
plt.ylabel('y')
plt.show()