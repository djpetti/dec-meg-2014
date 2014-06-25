#!/usr/bin/python

# Tests the CNN implementation with the MNIST classification problem.

import cPickle as pickle
import os

from theanets import Experiment
from theanets import feedforward
from weight_sharing import Cnn

import numpy as np

# Download MNIST pickle if we need to.
mnist_name = "mnist.pkl"
mnist_tar = mnist_name + ".gz"
mnist_url = "http://deeplearning.net/data/mnist/" + mnist_tar
if not os.path.exists(mnist_name):
  if not os.path.exists(mnist_tar):
    os.system("wget " + mnist_url)
  os.system("gunzip " + mnist_tar)

# Load it into memory.
print "Loading dataset..."
digits = pickle.load(file(mnist_name))

train = digits[0][0]
valid = digits[1][0]

# Cast "answer" data, because theano complains about it.
train_answers = digits[0][1].astype(np.int32)
valid_answers = digits[1][1].astype(np.int32)

# Make the classifier network. (Some of the hyperparameters here are arbitrary.)
experiment = Experiment(Cnn,
    img_shape = (10, 1, 28, 28),
    filter_shapes = ([1, 1, 9, 9],),
    layers = (50, 10),
    activation = "sigmoid")

print "Training classifier..."
# For some reason, size works here but batch_size doesn't.
experiment.add_dataset("train", (train, train_answers), size = 10)
experiment.add_dataset("valid", (valid, valid_answers), size = 10)
# Theanets has a bug that requires us to do this manually.
experiment.add_dataset("cg", digits[0], size = 10)
experiment.run()
print "Done!"

# Save it for later use.
experiment.save("CNN.pkl")
