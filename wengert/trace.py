"""
Computational graph building for a reverse-mode automatic
differentiation library, following autograd. Implemented and
annotated for personal pedogogical purposes.
"""
from contextlib import contextmanager

class Tracer:
    """
    Trace stack (effects handler) for the dynamic computational
    graph.
    """
    def __init__(self):
        self.top = -1
    
    @contextmanager
    def new_trace_frame(self):
        self.top += 1
        yield self.top
        self.top -= 1

class Node:
    """
    A node in the computational graph. Contains a wrapped value,
    application function, and list of pointers to parent nodes.
    """
    def __init__(self, value, fn, args, kwargs, parent_argnums, parents):
        """
        Node stores the following:
            value = output value of function
            fn    = function/operation that was just applied
            (args, kwargs) = inputs to the function in node
            parents = list of parent nodes
        """
        self.value   = value
        self.fn      = fn
        self.inputs  = (args, kwargs)
        self.parents = (parent_argnums, parents)

def primitive(fn_raw):
    """
    The edges of the computational graph is given by primitive (numpy)
    operations, wrapped so that important information such as gradients
    can be propagated. Computing between edges requires unboxing and
    reboxing the Node values. Implemented as a function decorator.
    """
    @wraps(fn_raw)
    def fn_wrapped(*args, **kwargs):
        pass