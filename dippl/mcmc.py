"""
Chpt 6: Markov Chain Monte Carlo
"""
import numpy as np
import math

from pyppl.distributions import *
from pyppl.viz import *

# Let's consider a random walk over the space of execution traces of a computation.
bernoulli = Bernoulli(p=0.5)

def cpsBinomial(k):
    _sample(
        lambda a:
            _sample(
                lambda b:
                    _sample(
                        lambda c:
                            k(a + b + c),
                        bernoulli),
                bernoulli),
        bernoulli)

trace = []
iterations = 1000

def _sample(cont, dist):
    val = dist.sample()
    trace.append({
        "k"   : cont,
        "val" : val,
        "dist": dist})
    cont(val)

returnHist = {}

def exit(val):
    global iterations
    global trace
    if val in returnHist:
        returnHist[val] += 1
    else:
        returnHist[val] = 1
    if iterations > 0:
        print(iterations)
        iterations -= 1
        # Choose a new proposal.
        regenFrom = np.random.randint(len(trace))
        regen     = trace[regenFrom]
        trace = trace[:regenFrom]
        _sample(regen["k"], regen["dist"])

def randomWalk(cpsComp):
    cpsComp(exit)
    # normalize
    norm = 0
    for v in returnHist:
        norm += v
    for v in returnHist:
        returnHist[v] /= norm
    return returnHist

print(randomWalk(cpsBinomial))
        