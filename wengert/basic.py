"""
Really basic implementation of a Wengert list
"""

# Wengert lists are lists of tuples (z, g, (y1,...)) where
#   z = output argument
#   g = operation
#   (y1,...) = input arguments
test = [
    ("z1", "add", ["x1", "x1"]),
    ("z2", "add", ["z1", "x2"]),
    ("f", "square", ["z2"])]

# Hash table to store function representations of operator names
G = {
    "add"   : lambda a, b: a + b,
    "square": lambda a: a*a}

# Hash table to store initialization values
val = {
    "x1": 3,
    "x2": 7}

print("x1 -> {}, x2 -> {}".format(val["x1"], val["x2"]))

# Evaluation function
def eval(f, val):
    for z, g, inputs in f:
        # Fetch operation
        op = G[g]
        # Apply to values
        args = list(map(lambda v: val[v], inputs))
        val[z] = op(*args)
    return val[z]

print("eval(test) -> {}".format(eval(test, val)))

# To do backpropagation of the Wengert list, we need the derivatives
# of each of the primitive operators.
DG = {
    "add"   : [(lambda a, b: 1), (lambda a, b: 1)],
    "square": [lambda a: 2*a]}

# We then go through the Wengert list in reverse, accumulating gradients
# when we pass through an operation.
def backpropagation(f, vals):
    # Initialize gradient tape
    delta = {"f": 1} # df/df = 1
    # Go through Wengert list in reverse order
    for z, g, inputs in reversed(f):
        args = list(map(lambda v: vals[v], inputs))
        for i in range(len(inputs)):
            # Get gradient
            dop = DG[g][i]
            yi  = inputs[i]
            if yi not in delta:
                delta[yi] = 0
            # Apply chain rule
            delta[yi] += delta[z] * dop(*args)
    return delta

delta = backpropagation(test, val)
print("df/dx1 -> {}, df/dx2 -> {}".format(delta["x1"], delta["x2"]))
