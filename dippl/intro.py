"""
Chpt 2: The PyPLL language.
"""
import math
from pyppl.distributions import *
from pyppl.viz import *

# A boring function.
def foo(x):
    bar = math.exp(x)
    baz = [] if x == 0 else [math.log(bar), foo(x-1)]
    return baz

print("foo(5) -> ", foo(5))

# PyPPL also has primitives for sampling distributions.
print("flip(0.5) -> ", flip(0.5))
viz(lambda: flip(0.5))

# As a universal PPL, we can construct stochastically recursive functions.
def geom(p):
    return 1 + geom(p) if flip(p) else 1

print("geom(0.5) -> ", geom(0.5))

# However, the most important feature is the ability to perform inference.
