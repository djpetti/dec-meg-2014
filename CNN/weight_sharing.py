#!/usr/bin/python

from operator import add, sub, floordiv, mul

import copy

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
    flat_size = reduce(mul, self.layer_shapes[-1], 1)
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
    self._inputs = TT.dtensor4("inputs")
    layer_outputs = self._inputs
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

    # Concatenate output maps into one long vector and set it as the input for
    # the normal part of our network.
    self.x = TT.flatten(layer_outputs)

  def _compile(self):
    self._compute = theano.function([self._inputs], self.hiddens + [self.y])

  def params(self, *args, **kwargs):
    params = super(Cnn, self).params(*args, **kwargs)
    params.extend(self.conv_weights)
    params.extend(self.conv_biases)
    return params
