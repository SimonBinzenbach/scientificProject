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