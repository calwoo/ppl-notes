"""
Coroutines: functions that receive continuations.
"""
import math

from pyppl.distributions import *
from pyppl.viz import *
from pyppl.utils import *


bernoulli = Bernoulli(p=0.5)

def binomial():
    a = sample(bernoulli)
    b = sample(bernoulli)
    c = sample(bernoulli)
    return a + b + c

# CPS version:
def cpsSample(k, dist):
    k(sample(dist))

def cpsBinomial(k):
    cpsSample(
        lambda a:
            cpsSample(
                lambda b:
                    cpsSample(
                        lambda c:
                            k(a + b+ c),
                        bernoulli),
                bernoulli),
        bernoulli)

cpsBinomial(print)
viz(lambda: binomial())

# Now we'll rewrite this so that `sample` gets the continuation of the point
# where it is called, and keeps going by calling this continuation.
# Such a structure is called a coroutine.
unexploredFutures = []
currScore = 0

def _sample(cont, dist):
    # Complete enumeration of executions.
    sup = dist.support()
    # Save all continuations of current branch, along with weighted score.
    list(map(lambda s:
        unexploredFutures.append({
            "k"    : lambda: cont(s),
            "score": currScore + dist.score(s)}), sup))
    # Resume next continuation.
    runNext()

def runNext():
    global currScore
    # Grab a continuation and "reset" the stack.
    next = unexploredFutures.pop(0)
    currScore = next["score"]
    next["k"]()

returnHist = {}

def exit(val):
    returnHist[val] = math.exp(currScore)
    if len(unexploredFutures) > 0:
        runNext()

def explore(cpsComp):
    cpsComp(exit)
    return returnHist

# New cpsBinomial
def cpsBinomialExplorer(k):
    _sample(
        lambda a:
            _sample(
                lambda b:
                    _sample(
                        lambda c:
                            k(a + b+ c),
                        bernoulli),
                bernoulli),
        bernoulli)

print("explore(cpsBinomial) -> {}".format(explore(cpsBinomialExplorer)))
