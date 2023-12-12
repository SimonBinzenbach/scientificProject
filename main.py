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


train_data, val_data = train_data.map(preprocess), val_data.map(preprocess) # using preprocess funtion to flatten out data matrixes

# activation functions
reLUls = tf.linspace(-2, 2, 15)  # ReLU(X)
reLUls = tf.cast(reLUls, tf.float32)
reLU = tf.nn.relu(reLUls)
# plt.plot(reLUls, reLU)

softmaxls = tf.linspace(-4, 4, 15)  # Softmax
softmaxls = tf.cast(softmaxls, tf.float32)
softmax = tf.nn.softmax(softmaxls, axis=0)
# plt.plot(softmaxls, softmax)

sigmoidls = tf.linspace(-5, 5, 15)  # Sigmoid
sigmoidls = tf.cast(sigmoidls, tf.float32)
sigmoid = tf.nn.sigmoid(sigmoidls)
# plt.plot(sigmoidls, sigmoid)

plt.show()
