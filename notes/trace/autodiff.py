import pyro
import pyro.distributions as dist
from pyro.poutine import trace
from pyro import sample, condition
import matplotlib.pyplot as plt

import torch
import torch.nn

"""
Autodifferentiation and backpropagation.
"""

# Suppose we observe x = 5 on a normal distribution and want to 
# determine the mean (mu) that gives us this.
mu = pyro.param("mu", torch.tensor(0.0))
norm = dist.Normal(mu, 1)
x = sample("x", norm)

# So we use standard gradient descent to maximize the likelihood of
# observing x = 5
prob = norm.log_prob(5)
print("before update: ", prob.item())

prob.backward()
# Gradient ascent update
mu.data += mu.grad
print("after update: ", dist.Normal(mu, 1).log_prob(5).item())

"""
We can generalize this to traces.
"""
def model():
    mu = pyro.param("mu", torch.tensor(0.0))
    norm = dist.Normal(mu, 1)
    return sample("x", norm)

# Reset
model()
mu = torch.tensor(0.0)
# Condition on observed data
conditional_model = condition(model, {"x": 5})

optimizer = torch.optim.Adam([pyro.param("mu")], lr=0.01)
losses = []

for _ in range(1000):
    # Traces contain the "likelihood" of the observed data as a trace
    tr = trace(conditional_model).get_trace()
    # Backpropagating the gradients from the likelihood trace
    prob = -tr.log_prob_sum()
    prob.backward()

    # Update
    optimizer.step()
    optimizer.zero_grad()
    losses.append(prob.item())
    print(pyro.param("mu"))

plt.plot(losses)
plt.title("Losses of Adam optimizer")
plt.xlabel("step")
plt.ylabel("negative log prob")
plt.show()


