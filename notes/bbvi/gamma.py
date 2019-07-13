import numpy as np
from scipy.special import gammaln, digamma
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
np.random.seed(11)

### DATA
# hyperparameters
K    = 12
N    = 1000
a_0  = 0.1
mu_0 = 5
# for k = 1..K, sample mu_k ~ Gamma(a_0, mu_0/a_0)
mu   = np.random.gamma(a_0, mu_0/a_0, size=K)
# for n = 1..N, k = 1..K, sample x_nk ~ Normal(mu_k, 1)
x    = np.random.normal(mu, 1, size=(N,K))

### DISTRIBUTIONS
class Gamma:
    def log_pdf(self, x, a, m):
        p = a*np.log(a) - a*np.log(m) - gammaln(a) + \
            (a-1)*np.log(x) - (a*x)/m
        return p
    
    def grad_a(self, x, a, m):
        grad = np.log(a) + 1 - np.log(m) - digamma(a) + \
            np.log(x) - x/m
        return grad

    def grad_m(self, x, a, m):
        return -a/m + (a*x)/m**2

class SoftPlus:
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def grad(self, x):
        return np.exp(x) / (1 + np.exp(x))

    def inv(self, x):
        return np.log(np.exp(x) - 1)

def log_normal(x, mu):
    return -0.5 * (x-mu)**2 - np.log(np.sqrt(2*np.pi))

gamma = Gamma()
sp = SoftPlus()

### BBVI
# number of samples for each monte carlo estimate
num_samples = 1024
# initialize variational parameters
lam_a  = 0.1 * np.ones(K)
lam_mu = 0.01 * np.ones(K)

# run BBVI for some iterations
num_epochs = 1500
mu_log = []
for epoch in tqdm(range(num_epochs)):
    # draw sample means for monte carlo estimate
    sample_mus = np.random.gamma(sp(lam_a), sp(lam_mu)/sp(lam_a), size=(num_samples,K))
    ## GUARD: prevent divide-by-zero errors
    sample_mus[sample_mus < 1e-300] = 1e-300
    # get prior prob and variational prob
    p = gamma.log_pdf(sample_mus, a_0, mu_0)
    q = gamma.log_pdf(sample_mus, sp(lam_a), sp(lam_mu))
    grad_a = sp.grad(lam_a) * gamma.grad_a(sample_mus, sp(lam_a), sp(lam_mu))
    grad_m = sp.grad(lam_mu) * gamma.grad_m(sample_mus, sp(lam_a), sp(lam_mu))
    # probability of observations
    for i in range(N):
        p += log_normal(x[i], sample_mus)
    
    # robbins-monro sequence for step size
    rho = (epoch + 1024) ** -0.7
    # update variational parameters
    lam_a  += rho * np.mean(grad_a*(p-q), axis=0)
    lam_mu += rho * np.mean(grad_m*(p-q), axis=0)
    ## GUARD: prevent divide-by-zero errors and overflow
    lam_a[lam_a < sp.inv(5e-3)]     = sp.inv(5e-3)
    lam_mu[lam_mu < sp.inv(1e-5)]   = sp.inv(1e-5)
    lam_a[lam_a > sp.inv(np.log(sys.float_info.max))]   = sp.inv(np.log(sys.float_info.max))
    lam_mu[lam_mu > sp.inv(np.log(sys.float_info.max))] = sp.inv(np.log(sys.float_info.max))
    # add to log
    mu_log.append(np.copy(sp(lam_mu)))

for i in range(K):
    ax = plt.subplot(4,3,i+1)
    ax.plot(list(map(lambda x: float(x[i]), mu_log)))
    ax.hlines(mu[i], xmin=0, xmax=num_epochs)
plt.show()