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
    preprocess)  # using preprocess funtion to flatten out data matrixes

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


# Denselayer
class DenseLayer(tf.Module):
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


plt.show()
