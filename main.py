import math

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile
import os
import tensorflow as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import tensorflow_datasets as tfds


matplotlib.rcParams['figure.figsize'] = [9, 6]
tf.random.set_seed(42)


def xavier_init(shape):
    in_dim, out_dim = shape
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev)


def calculate_edge(x, y, image, height, width):
    strongest_edge = 0.0
    for sx in range(-1, 2):
        for sy in range(-1, 2):
            if sx == 0 and sy == 0:
                continue

            nx, ny = x + sx, y + sy
            if 0 <= nx < height and 0 <= ny < width:
                edge = np.linalg.norm(image[x, y] - image[nx, ny])
                strongest_edge = max(strongest_edge, edge)

    return x, y, strongest_edge / (math.sqrt(256**2 * 3))  # normalize


def omnidirectionalEdgeMapColor(image):

    width = image.shape[0]
    height = image.shape[1]
    new_tensor = np.zeros(shape=(height, width))

    with ThreadPoolExecutor() as executor:
        futures = []
        for x in range(height):
            for y in range(width):
                futures.append(executor.submit(calculate_edge, x, y, image, height, width))

        for future in futures:
            x, y, edge_value = future.result()
            new_tensor[x, y] = edge_value
            if x % 100 == 0 and y % 100 == 0:
                print(f"x: {x} y: {y} edge: {new_tensor[x, y]}")

    return new_tensor


def max_pooling(input_tensor):
    input_height, input_width, input_channels = input_tensor.shape

    output_height = input_height // 2
    output_width = input_width // 2
    output_channels = input_channels

    output_tensor = tf.zeros((output_height, output_width, output_channels), dtype=tf.float32)

    for y in range(0, input_height, 2):
        for x in range(0, input_width, 2):
            pool_region = input_tensor[y:y+2, x:x+2, :]
            pooled_value = tf.reduce_max(pool_region, axis=(0, 1))
            output_tensor[y//2, x//2, :] = pooled_value

    return output_tensor


def preprocess(x):  # function for flattening out feature matrix
    x = tf.reshape(x, shape=[-1, 40000])
    x = x / 255
    return x


# Activation Function showcases
reLU_ls = tf.linspace(-2, 2, 15)  # ReLU(X)
reLU_ls = tf.cast(reLU_ls, tf.float32)
reLU = tf.nn.relu(reLU_ls)

softmax_ls = tf.linspace(-4, 4, 15)  # Softmax
softmax_ls = tf.cast(softmax_ls, tf.float32)
softmax = tf.nn.softmax(softmax_ls, axis=0)

sigmoid_ls = tf.linspace(-5, 5, 15)  # Sigmoid
sigmoid_ls = tf.cast(sigmoid_ls, tf.float32)
sigmoid = tf.nn.sigmoid(sigmoid_ls)

tanh_ls = tf.linspace(-5, 5, 15)  # Tanh
tanh_ls = tf.cast(tanh_ls, tf.float32)
tanh = tf.nn.tanh(tanh_ls)


def plot_metrics(train_metric, val_metric, metric_type):  # Visualize metrics vs training Epochs
    plt.figure()
    plt.plot(range(len(train_metric)), train_metric, label=f"Training {metric_type}")
    plt.plot(range(len(val_metric)), val_metric, label=f"Validation {metric_type}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_type)
    plt.legend()
    plt.title(f"{metric_type} vs Training epochs")


# MLP Initialization stuff
class DenseLayer(tf.Module):  # Function to initialize DenseLayer
    def __init__(self, out_dim, weight_init=xavier_init, activation=tf.identity):
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


class ConvolutionLayer(tf.Module):  # Function to initialize DenseLayer
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        out_tensor = omnidirectionalEdgeMapColor(x)
        out_tensor = max_pooling(out_tensor)
        out_tensor = preprocess(out_tensor)
        return out_tensor


class CNN(tf.Module):  # Function to initialize MLP
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


def binary_cross_entropy(y_pred, y):
    epsilon = 1e-7
    return -tf.reduce_mean(y * tf.math.log(y_pred + epsilon) + (1 - y) * tf.math.log(1 - y_pred + epsilon))


def accuracy(y_pred, y):  # accuracy function
    class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)
    is_equal = tf.equal(y, class_preds)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))


# Training Functions
def train_step(x_batch, y_batch, loss, acc, model, optimizer):  # Model Update
    with tf.GradientTape() as tape:
        y_pred = model(x_batch)
        batch_loss = loss(y_pred, y_batch)
    batch_acc = acc(y_pred, y_batch)
    grads = tape.gradient(batch_loss, model.variables)
    optimizer.apply_gradients(grads, model.variables)
    return batch_loss, batch_acc


def val_step(x_batch, y_batch, loss, acc, model):  # Evaluate Model
    y_pred = model(x_batch)
    batch_loss = loss(y_pred, y_batch)
    batch_acc = acc(y_pred, y_batch)
    return batch_loss, batch_acc


def train_model(mlp, train_data, val_data, loss, acc, optimizer, epochs):  # Training
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(epochs):
        batch_losses_train, batch_accs_train = [], []
        batch_losses_val, batch_accs_val = [], []

        for x_batch, y_batch in train_data:
            batch_loss, batch_acc = train_step(x_batch, y_batch, loss, acc, mlp, optimizer)
            batch_losses_train.append(batch_loss)
            batch_accs_train.append(batch_acc)

        for x_batch, y_batch in val_data:
            batch_loss, batch_acc = val_step(x_batch, y_batch, loss, acc, mlp)
            batch_losses_val.append(batch_loss)
            batch_accs_val.append(batch_acc)

        train_loss, train_acc = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train)
        val_loss, val_acc = tf.reduce_mean(batch_losses_val), tf.reduce_mean(batch_accs_val)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Epoch: {epoch}")
        print(f"Training loss: {train_loss:.3f}, Training accuracy: {train_acc:.3f}")
        print(f"Validation loss: {val_loss:.3f}, Validation accuracy: {val_acc:.3f}")
    return train_losses, train_accs, val_losses, val_accs


#   Exporting Functions
class ExportModule(tf.Module):
    def __init__(self, model, preprocess, class_pred):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.class_pred = class_pred

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.uint8)])  # New Data
    def __call__(self, x):
        x = self.preprocess(x)
        y = self.model(x)
        y = self.class_pred(y)
        return y


def preprocess_test(x):  # Get Raw Data
    x = tf.reshape(x, shape=[-1, 784])
    x = x / 255
    return x


def class_predict(y):  # Predict
    return tf.argmax(tf.nn.softmax(y), axis=1)


def accuracy_score(y_pred, y):  # Compare and calc Accuracy
    is_equal = tf.equal(y_pred, y)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))


# importiertes dataset
train_data, val_data, test_data = tfds.load("cats_vs_dogs",
                                            split=['train[10000:20000]', 'train[0:10000]', 'train[20000:]'],
                                            batch_size=128, as_supervised=True)


# Initialization of CNN
hidden_layer_1_size = 700
hidden_layer_2_size = 500
output_size = 2
cnn_model = CNN([
    ConvolutionLayer(),
    DenseLayer(out_dim=hidden_layer_1_size, activation=tf.nn.relu),
    DenseLayer(out_dim=hidden_layer_2_size, activation=tf.nn.sigmoid),
    DenseLayer(out_dim=output_size)])
# Train Loop
train_losses, train_accs, val_losses, val_accs = train_model(cnn_model, train_data, val_data,
                                                             loss=binary_cross_entropy, acc=accuracy,
                                                             optimizer=Adam(), epochs=3)


# Init Export Module
cnn_model_export = ExportModule(model=mlp_model,
                                preprocess=preprocess_test,
                                class_pred=class_predict)
# Export
models = tempfile.mkdtemp()
save_path = os.path.join(models, 'mlp_model_export')
tf.saved_model.save(mlp_model_export, save_path)

# Load
cnn_loaded = tf.saved_model.load(save_path)
x_test, y_test = test_data[-1]
test_classes = mlp_loaded(x_test)
test_acc = accuracy_score(test_classes, y_test)
print(f"Test Accuracy: {test_acc:.3f}")

# plotting
# plt.plot(reLU_ls, reLU)
# plt.plot(softmax_ls, softmax)
# plt.plot(sigmoid_ls, sigmoid)
# plt.plot(tanh_ls, tanh)
# plot_metrics(train_losses, val_losses, "cross entropy loss")
# plot_metrics(train_accs, val_accs, "accuracy")

plt.show()
