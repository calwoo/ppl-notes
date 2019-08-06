"""
In the interview process for each student, the student flips a coin, 
hidden from the interviewer. The student agrees to answer honestly 
if the coin comes up heads. Otherwise, if the coin comes up tails, 
the student (secretly) flips the coin again, and answers 
"Yes, I did cheat" if the coin flip lands heads, and "No, I did 
not cheat", if the coin flip lands tails. This way, the interviewer 
does not know if a "Yes" was the result of a guilty plea, or a Heads 
on a second coin toss. Thus privacy is preserved and the researchers 
receive honest answers.
"""

import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import theano.tensor as tt

N = 100
X = 35
with pm.Model() as model:
    p = pm.Uniform("freq_cheating", 0, 1)
    true_answers = pm.Bernoulli("truths", p, shape=N, testval=np.random.binomial(1, 0.5, N))
    first_coin_flips = pm.Bernoulli("first_flips", 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))
    second_coin_flips = pm.Bernoulli("second_flips", 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))
    val = first_coin_flips * true_answers + (1-first_coin_flips)*second_coin_flips
    observed_proportion = pm.Deterministic("observed_proportion", tt.sum(val)/float(N))
    observations = pm.Binomial("obs", N, observed_proportion, observed=X)

    step = pm.Metropolis(vars=[p])
    trace = pm.sample(40000, step=step)
    burned_trace = trace[15000:]

plt.figsize(12.5, 3)
p_trace = burned_trace["freq_cheating"][15000:]
plt.hist(p_trace, histtype="stepfilled", normed=True, alpha=0.85, bins=30, 
         label="posterior distribution", color="#348ABD")
plt.vlines([.05, .35], [0, 0], [5, 5], alpha=0.3)
plt.xlim(0, 1)
plt.legend()