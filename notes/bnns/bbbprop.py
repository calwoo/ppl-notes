import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model
plt.style.use("seaborn-pastel")

### DATA
def fn(x, sigma):
    eps = np.random.randn(*x.shape) * sigma
    return 10 * np.sin(2*np.pi*x) + eps

# Data parameters
num_data = 32
noise = 1.0
# Data
X = np.linspace(-0.5, 0.5, num_data).reshape(-1,1)
y = fn(X, noise)
y_groundtruth = fn(X, 0.0)

plt.scatter(X, y, marker='+', label='Training data')
plt.plot(X, y_groundtruth, label='Truth')
plt.title('Noisy training data and ground truth')
plt.legend()
# plt.show()

### BAYESIAN LINEAR LAYER
class DenseVariational(Layer):
    """
    Implementation of a Bayesian dense layer for BNNs
    with Gaussian weights. We use the reparamterization
    trick to allow backpropagation to update the weight
    parameters.
    """
    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        self.weight_mu = self.add_weight(name="weight_mu",
                                         shape=[int(input_shape[-1]), self.output_dim],
                                         initializer=tf.keras.initializers.normal(stddev=0.01),
                                         trainable=True)
        self.weight_rho = self.add_weight(name="weight_rho",
                                          shape=(input_shape[-1], self.output_dim),
                                          initializer=tf.keras.initializers.constant(0.0),
                                          trainable=True)
        self.bias_mu = self.add_weight(name="bias_mu",
                                       shape=(self.output_dim,),
                                       initializer=tf.keras.initializers.normal(stddev=0.01),
                                       trainable=True)
        self.bias_rho = self.add_weight(name="bias_rho",
                                        shape=(self.output_dim,),
                                        initializer=tf.keras.initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, x):
        # Sample weight
        weight_sigma = tf.math.softplus(self.weight_rho)
        weight = self.weight_mu + weight_sigma * tf.random.normal(self.weight_mu.shape)
        # Sample bias
        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
        # Get logit
        logit = tf.matmul(x, weight) + bias
        return self.activation(logit)


### MODEL
input_layer = Input(shape=(1,))
x = DenseVariational(20, activation="relu")(input)
x = DenseVariational(20, activation="relu")(x)
x = DenseVariational(1)(x)
output_layer = x

model = Model(input_layer, output_layer)

print(model(X))

