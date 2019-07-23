import autograd.numpy as np
from autograd import grad

# Black-box variational inference
def BBVI(log_p, num_params, num_samples):
    """
    Implementation of black-box variational inference with
    variational family given by normal distributions with
    diagonal covariance.
        log_p = log probability of model, p(x,z)
        num_params = number of latent variables in model
        num_samples = number of Monte Carlo estimates to use
    """
    def entropy(mu, log_sigma):
        """
        Computes the differential entropy of a multivariate
        Gaussian.
        """
        return 0.5 * num_params * (1.0 + np.log(2*np.pi)) + np.sum(log_sigma)

    def ELBO(variational_params):
        """
        Computes the evidence lower bound of the variational inference
        procedure. Recall that the ELBO = entropy + "energy".
        """
        # Get variational distribution q(z)
        mu, log_sigma = variational_params[:num_params], variational_params[num_params:]
        # Monte Carlo samples of the latent variables
        latent_samples = np.exp(log_sigma) * np.random.randn(num_samples, num_params) + mu
        # ELBO = entropy + "energy"
        elbo = entropy(mu, log_sigma) + np.mean(log_p(latent_samples))
        return -elbo

    gradient = grad(ELBO)
    return ELBO, gradient
