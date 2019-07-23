import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

### DATA
n, k = 100, 3
X = np.random.normal(size=(n, k))
w = np.random.normal(size=(k, 1)) * 5
y = np.random.normal(X.dot(w), 1.0)

### TF MODEL
model_X = tf.placeholder(tf.float64, shape=[None, k])
model_y = tf.placeholder(tf.float64, shape=[None, 1])

model_noise_prec = tf.placeholder(tf.float64, shape=[])
model_w_prec = tf.placeholder(tf.float64, shape=[])

# compute mean/var of posterior
model_I = tf.eye(k, dtype=tf.float64)
model_Lambda = model_noise_prec * tf.matmul(model_X, model_X, transpose_a=True) + model_w_prec * model_I
model_L = tf.linalg.cholesky(model_Lambda)
model_L_inv = tf.matrix_triangular_solve(model_L, I)
model_mu = model_noise_prec * tf.cholesky_solve(model_L, tf.matmul(model_X, model_y, transpose_a=True))

with tf.Session() as sess:
    feed_dict = {
        model_X: X,
        model_y: y,
        model_noise_prec: 1.0,
        model_w_prec: 0.001}
    mu = sess.run(model_mu, feed_dict=feed_dict)
    print("computed mean: {}".format(mu.ravel()))
    print("true mean    : {}".format(w.ravel()))



if False:
    plt.figure(figsize=(17,6))
    plt.plot(X.ravel(), y.ravel(), ".")
    plt.grid()
    plt.show()