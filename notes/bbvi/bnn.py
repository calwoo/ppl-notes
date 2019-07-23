import autograd.numpy as np
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt

from bbvi import BBVI

### DATA
def dataset(num_data=20, noise=0.5):
    num_dims = 1
    inputs  = np.concatenate([np.linspace(0, 2, num=num_data/2),
                              np.linspace(6, 8, num=num_data/2)])
    targets = np.cos(inputs) + np.random.randn(num_data) * noise
    inputs = (inputs - 4.0) / 4.0
    inputs  = inputs.reshape((len(inputs), num_dims))
    targets = targets.reshape((len(targets), num_dims))
    return inputs, targets

### BAYESIAN NEURAL NET
def bayesian_neural_net(layers, L2_reg, w_var, activation=np.tanh):
    """
    Implementation of a Bayesian neural net.
        layers = list of integers, indicates number of hidden units
                 per layer.
        L2_reg = L2 regularization weight.
        w_var  = shared variance of each weight.
        activation = activation function for each hidden layer
    """
    # List of shapes
    shapes = list(zip(layers[:-1], layers[1:]))
    # Total number of weights -> m*n + n where the n comes from the bias
    num_weights = sum((m+1)*n for m, n in shapes)
    
    # Generator for weight extraction
    def layers_gen(weights):
        num_posterior_samples = len(weights)
        for m, n in shapes:
            # Yield weights and biases
            ws = weights[:, :m*n]     .reshape((num_posterior_samples, m, n))
            bs = weights[:, m*n:m*n+n].reshape((num_posterior_samples, 1, n))
            yield ws, bs
            weights = weights[:, (m+1)*n:]

    # Forward pass
    def forward(weights, inputs):
        """
        Forward propagation of neural network.
            weights = matrix of shape (num_posterior_samples, num_weights)
            inputs  = matrix of shape (num_data, num_dims)
        """
        inputs = np.expand_dims(inputs, 0)
        for weight, bias in layers_gen(weights):
            # weight has shape (num_posterior_samples, m, n)
            # bias has shape (num_posterior_samples, 1, n)
            logits = np.einsum("mnd,mdo->mno", inputs, weight) + bias
            # sidenote: Einstein summation is insanely cool.
            inputs = activation(logits)
        return logits

    # Log probability density
    def logp(weights, inputs, targets):
        log_prior = -L2_reg * np.sum(weights**2, axis=1)
        predictions = forward(weights, inputs)
        log_likelihood = -np.sum((predictions-targets)**2, axis=1)[:, 0] / w_var
        return log_prior + log_likelihood

    return num_weights, forward, logp


if __name__ == "__main__":
    # Get dataset
    inputs, targets = dataset(num_data=40, noise=0.1)
    # Create BNN
    relu = lambda x: np.maximum(0, x)
    rbf  = lambda x: np.exp(-x**2)
    num_weights, forward, logp = bayesian_neural_net(
        layers=[1, 20, 20, 1],
        L2_reg=0.1,
        w_var =0.01,
        activation=relu)
    # Log probability density of model
    log_joint = lambda weights: logp(weights, inputs, targets)
    # Extract gradients for BBVI
    ELBO, gradient = BBVI(log_joint, num_weights, num_samples=20)
    # Set up plotting
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    ### Perform BBVI
    def lr_scheduler(init_lr=0.01, decay=0.99):
        # Generator for lr
        lr = init_lr
        while True:
            yield lr
            lr *= decay

    sched = lr_scheduler(init_lr=0.1, decay=1.0)
    num_epochs = 1000
    # Adam optimizer parameters
    beta1, beta2 = 0.9, 0.999
    epsilon = 10e-8
    m = np.zeros(2 * num_weights)
    v = np.zeros(2 * num_weights)
    # Set initial parameters
    initial_mean = np.random.randn(num_weights)
    initial_log_sigma = -5 * np.ones(num_weights)
    initial_variational_params = np.concatenate([initial_mean, initial_log_sigma])
    # Optimize
    print("-> Optimizing variational parameters...")
    print("-> Initial ELBO: {}".format(ELBO(initial_variational_params)))
    vparams = initial_variational_params
    for epoch in range(num_epochs):
        lr = next(sched)
        ### Adam optimizer
        g = gradient(vparams)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*(g**2)
        # Correcting biased terms
        mhat = m / (1-beta1**(epoch+1))
        vhat = v / (1-beta2**(epoch+1))
        # Update step
        vparams -= lr * mhat / (np.sqrt(vhat) + epsilon)

        ### Logging and sampling from posterior
        print("Epoch {} -> ELBO: {}".format(epoch, ELBO(vparams)))
        # Sample from posterior
        num_posterior_samples = 10
        mu, log_sigma = vparams[:num_weights], vparams[num_weights:]
        posterior_samples = mu + np.exp(log_sigma) * np.random.randn(num_posterior_samples, num_weights)

        plot_inputs = np.linspace(-8, 8, num=400)
        outputs = forward(posterior_samples, np.expand_dims(plot_inputs, 1))
        # Plot
        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'bx')
        ax.plot(plot_inputs, outputs[:, :, 0].T)
        ax.set_ylim([-2, 3])
        plt.draw()
        plt.pause(1.0/60.0)