#!/usr/bin/python

from operator import add, sub, floordiv, mul

import copy
import cPickle as pickle
import gzip

from theanets import feedforward
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy as np
import theano
import theano.tensor as TT

# A class for representing a neural network with weight sharing.
class Cnn(feedforward.Classifier):
  # There is no specified input layer for the fully connected part of the Cnn.
  # (Technically, there is, but it is implemented automatically.) Therefore, the
  # first item in "layers" is actually the size of the first hidden layer.
  def __init__(self, img_shape, filter_shapes, layers, activation,
      pool_sizes = "default", *args, **kwargs):
    self.img_shape = img_shape
    self.conv_weights = []
    self.conv_biases = []
    self.filter_shapes = []
    self.pool_sizes = []
    self.pool_mult_biases = []
    self.pool_add_biases = []

    # The CNN is designed to work with normal trainers, which generally
    # provide the input data for the network as a 2d matrix where each row is
    # one item in a minibatch, not in the 4d format that is more logical for
    # CNN's. If this option is set, the CNN will expect input in a flattenned 2d
    # form and automatically reshape it back to a 4d form.
    self.twod_inputs = kwargs.get("accept_2d_inputs", True)

    self.activation_func = self._build_activation(activation)

    if pool_sizes != "default":
      for shape, size in zip(filter_shapes, pool_sizes):
        self._add_layer_pair(shape, size)
    else:
      for shape in filter_shapes:
        self._add_layer_pair(shape, (2, 2))

    # Figure out and set the correct number of inputs.
    self.__find_shapes()
    layers = list(layers)
    flat_size = reduce(mul, self.layer_shapes[-1][1:], 1)
    layers.insert(0, flat_size)

    super(Cnn, self).__init__(layers, activation, *args, no_x = True, **kwargs)

  # Override so things get run in the correct order.
  def _create_forward_map(self, *args, **kwargs):
    self._make_graph()
    return super(Cnn, self)._create_forward_map(*args, **kwargs)

  def _add_layer_pair(self, filter_shape, pool_size):
    # Check for the same number of feature maps.
    if filter_shape[1] != self.img_shape[1]:
      raise RuntimeError("Number of input feature maps must be the same.")
    self.filter_shapes.append(filter_shape)

    # Initialize weight (kernel) values.
    fan_in = np.prod(filter_shape[1:])
    rng = np.random.RandomState()
    weight_values = np.asarray(rng.uniform(
        low = -np.sqrt(3. / fan_in),
        high = np.sqrt(3. / fan_in),
        size = filter_shape), dtype = theano.config.floatX)
    weights = theano.shared(value = weight_values, name = "weights")
    self.conv_weights.append(weights)

    # Initialize the biases.
    bias_values = np.zeros((filter_shape[0],), dtype = theano.config.floatX)
    biases = theano.shared(value = bias_values, name = "biases")
    self.conv_biases.append(biases)

    # Set up subsampling parameters.
    self.pool_sizes.append(pool_size)

  # Figure out the shapes of each convolution layer.
  def __find_shapes(self):
    self.layer_shapes = []
    next_shape = list(self.img_shape)
    self.layer_shapes.append(list(self.img_shape))
    
    for i in range(0, len(self.conv_weights)):
      filter_shape = list(self.filter_shapes[i])
      
      # Convolution.
      next_shape[2:] = map(sub, next_shape[2:], filter_shape[2:])
      next_shape[2:] = map(add, next_shape[2:], [1, 1])
      next_shape[1] = filter_shape[1]

      # Max pooling.
      next_shape[2:] = map(floordiv, next_shape[2:], self.pool_sizes[i])

      # The fun copying stuff is so new changes to next_shape don't modify
      # references already in the list.
      self.layer_shapes.append([])
      for num in next_shape:
        cop = copy.deepcopy(num)
        self.layer_shapes[-1].append(cop)

  # Builds a graph for the CNN.
  def _make_graph(self):
    if not self.twod_inputs:
      self._inputs = TT.dtensor4("inputs")
      layer_outputs = self._inputs
    else:
      self._inputs = TT.matrix("inputs")
      layer_outputs = self._inputs.reshape(self.img_shape)
    
    for i in range(0, len(self.conv_weights)):
      # Perform the convolution.
      conv_out = conv.conv2d(layer_outputs, self.conv_weights[i],
          filter_shape = self.filter_shapes[i],
          image_shape = self.layer_shapes[i])
      
      # Downsample the feature maps.
      pooled_out = downsample.max_pool_2d(conv_out, self.pool_sizes[i],
          ignore_border = True)

      # Account for the bias. Since it is a vector, we first need to reshape it
      # to (1, n_filters, 1, 1).
      layer_outputs = self.activation_func(pooled_out + \
          self.conv_biases[i].dimshuffle("x", 0, "x", "x"))

    # Concatenate output maps into one big matrix where each row is the
    # concatenation of all the feature maps from one item in the batch.
    next_shape = self.layer_shapes[i + 1]
    new_shape = (next_shape[0], reduce(mul, next_shape[1:], 1))
    print "New Shape: " + str(new_shape)
    self.x = layer_outputs.reshape(new_shape)

  def _compile(self):
    self._compute = theano.function([self._inputs], self.hiddens + [self.y])
  
  def params(self, *args, **kwargs):
    params = super(Cnn, self).params(*args, **kwargs)
    params.extend(self.conv_weights)
    params.extend(self.conv_biases)
    return params

  @property
  def inputs(self):
    return [self._inputs, self.k]

  # Save override that also saves CNN-specific parameters.
  def save(self, filename):
    opener = gzip.open if filename.lower().endswith(".gz") else open
    handle = opener(filename, "wb")
    
    params = {}
    params["weights"] = [param.get_value().copy() for param in self.weights]
    params["biases"] = [param.get_value().copy() for param in self.biases]
    params["conv_weights"] = \
        [param.get_value().copy() for param in self.conv_weights]
    params["conv_biases"] = \
        [param.get_value().copy() for param in self.conv_biases]

    pickle.dump(params, handle, -1)
    handle.close()

  # Load override that also loads CNN specific parameters.
  def load(self, filename):
    opener = gzip.open if filename.lower().endswith(".gz") else open
    handle = opener(filename, "rb")

    params = pickle.load(handle)

    for target, source in zip(self.weights, params["weights"]):
      target.set_value(source)
    for target, source in zip(self.biases, params["biases"]):
      target.set_value(source)
    for target, source in zip(self.conv_weights, params["conv_weights"]):
      target.set_value(source)
    for target, source in zip(self.conv_biases, params["conv_biases"]):
      target.set_value(source)

    handle.close()
    
