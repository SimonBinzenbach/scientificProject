import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile
import os
import tensorflow as tf
import tensorflow_datasets as tfds

matplotlib.rcParams['figure.figsize'] = [9, 6]
tf.random.set_seed(42)

# importiertes dataset
train_data, val_data, test_data = tfds.load("mnist",
                                            split=['train[10000:]', 'train[0:10000]', 'test'],
                                            batch_size=128, as_supervised=True)


def preprocess(x, y):  # function for flattening out feature matrix
    x = tf.reshape(x, shape=[-1, 784])
    x = x / 255
    return x, y


train_data, val_data = train_data.map(preprocess), val_data.map(
    preprocess)  # using preprocess funtion to flatten out data matrices

# activation functions
reLU_ls = tf.linspace(-2, 2, 15)  # ReLU(X)
reLU_ls = tf.cast(reLU_ls, tf.float32)
reLU = tf.nn.relu(reLU_ls)
# plt.plot(reLU_ls, reLU)

softmax_ls = tf.linspace(-4, 4, 15)  # Softmax
softmax_ls = tf.cast(softmax_ls, tf.float32)
softmax = tf.nn.softmax(softmax_ls, axis=0)
# plt.plot(softmax_ls, softmax)

sigmoid_ls = tf.linspace(-5, 5, 15)  # Sigmoid
sigmoid_ls = tf.cast(sigmoid_ls, tf.float32)
sigmoid = tf.nn.sigmoid(sigmoid_ls)
# plt.plot(sigmoid_ls, sigmoid)

tanh_ls = tf.linspace(-5, 5, 15)  # Tanh
tanh_ls = tf.cast(tanh_ls, tf.float32)
tanh = tf.nn.tanh(tanh_ls)
# plt.plot(tanh_ls, tanh)


# MLP Initialization stuff
class DenseLayer(tf.Module):  # Function to initialize DenseLayer
    def __init__(self, out_dim, weight_init=tf.keras.initializers.GlorotUniform(), activation=tf.identity):
        super().__init__()
        self.out_dim = out_dim
        self.weight_init = weight_init
        self.activation = activation
        self.built = False

    def __call__(self, x):
        if not self.built:
            self.in_dim = x.shape[1]
            self.w = tf.Variable(self.weight_init(shape=(self.in_dim, self.out_dim)))
            self.b = tf.Variable(tf.zeros(shape=(self.out_dim,)))
            self.built = True
        z = tf.add(tf.matmul(x, self.w), self.b)
        return self.activation(z)


class MLP(tf.Module):  # Function to initialize MLP
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    @tf.function
    def __call__(self, x, preds=False):
        for layer in self.layers:
            x = layer(x)
        return x


# Optimizers
class GradientDescent(tf.Module):  # GradientDescent
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.title = f"Gradient descent optimizer: learning rate={self.learning_rate}"

    def apply_gradients(self, grads, vars):
        # Update variables
        for grad, var in zip(grads, vars):
            var.assign_sub(self.learning_rate * grad)


class Adam(tf.Module):  # Adaptive moment estimation

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
        super().__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.ep = ep
        self.t = 1.
        self.v_dvar, self.s_dvar = [], []
        self.title = f"Adam: learning rate={self.learning_rate}"
        self.built = False

    def apply_gradients(self, grads, vars):
        if not self.built:
            for var in vars:
                v = tf.Variable(tf.zeros(shape=var.shape))
                s = tf.Variable(tf.zeros(shape=var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
            self.built = True
        for i, (d_var, var) in enumerate(zip(grads, vars)):
            self.v_dvar[i] = self.beta_1 * self.v_dvar[i] + (1 - self.beta_1) * d_var
            self.s_dvar[i] = self.beta_2 * self.s_dvar[i] + (1 - self.beta_2) * tf.square(d_var)
            v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1 ** self.t))
            s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2 ** self.t))
            var.assign_sub(self.learning_rate * (v_dvar_bc / (tf.sqrt(s_dvar_bc) + self.ep)))
        self.t += 1.


def cross_entropy_loss(y_pred, y):  # cross entropy loss function
    sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(sparse_ce)


def accuracy(y_pred, y):  # accuracy function
    class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)
    is_equal = tf.equal(y, class_preds)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))


# Initialization of MLP
hidden_layer_1_size = 700
hidden_layer_2_size = 500
output_size = 10
mlp_model = MLP([
    DenseLayer(out_dim=hidden_layer_1_size, activation=tf.nn.relu),
    DenseLayer(out_dim=hidden_layer_2_size, activation=tf.nn.relu),
    DenseLayer(out_dim=output_size)])


# Training Loop

plt.show()
