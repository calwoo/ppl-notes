"""
Chpt 3: Exploring the executions of a random computation.
"""

from pyppl.distributions import *
from pyppl.viz import *

### Continuations
# A continuation is a reificiation of a computation's future.
def square(x):
    return x * x

print("square(3) -> {}".format(square(3)))

# Continuation-passing style (CPS) is a way to write programs such that the current
# continuation is always explicit.
def cpsSquare(k, x):
    k(x * x)

print("cpsSquare(print, 3) -> ")
cpsSquare(print, 3)

# In CPS, functions never return.
def factorial(n):
    if n == 0:
        return 1
    else:
        return factorial(n-1) * n

print("factorial(5) -> {}".format(factorial(5)))

def cpsFactorial(k, n):
    if n == 0:
        k(1)
    else:
        cpsFactorial(
            lambda x: k(x * n),
            n-1)

print("cpsFactorial(print, 5) ->")
cpsFactorial(print, 5)

# This is effectively a tail-recursive form. Unfortunately, Python
# doesn't support TCO, so it doesn't really matter if things are written
# in this style.
def factorial2(n, acc):
    if n == 0:
        return acc
    else:
        return factorial2(n-1, n*acc)

# CPS version:
def cpsFactorial2(k, n, acc):
    if n == 0:
        k(acc)
    else:
        cpsFactorial2(k, n-1, n*acc)

print("factorial2(5, 1) -> {}".format(factorial2(5,1)))
print("cpsFactorial2(print, 5, 1) ->")
cpsFactorial2(print, 5, 1)

# CPS is useful because it allows us to reify the control flow. For example, it
# allows us to explicitly deal with exception handling.
def totalCPSFactorial(k, err, n):
    if n < 0:
        err("cpsFactorial: n < 0!")
    elif n == 0:
        k(1)
    else:
        totalCPSFactorial(
            lambda x: k(x * n),
            err,
            n-1)

def printError(x):
    print("Error: " + x)

totalCPSFactorial(print, printError, 5)
totalCPSFactorial(print, printError, -1)

