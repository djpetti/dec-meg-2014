#!/usr/bin/python

# Tests the CNN implementation with the MNIST classification problem.

from __future__ import division

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
test = digits[2][0]

# Cast "answer" data, because theano complains about it.
train_answers = digits[0][1].astype(np.int32)
valid_answers = digits[1][1].astype(np.int32)
test_answers = digits[2][1].astype(np.int32)

# Make the classifier network. (Some of the hyperparameters here are arbitrary.)
if not os.path.exists("CNN.pkl"):
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

print "Testing network..."
cnn = Cnn(img_shape = (test.shape[0], 1, 28, 28),
      filter_shapes = ([1, 1, 9, 9],),
      layers = (50, 10),
      activation = "sigmoid")
cnn.load("CNN.pkl")

result = cnn.classify(test)

correct = 0
incorrect = 0
for i in range(0, test.shape[0]):
  if result[i] == test_answers[i]:
    correct += 1
  else:
    incorrect += 1

print "%f percent accuracy." % (correct / (correct + incorrect) * 100)
