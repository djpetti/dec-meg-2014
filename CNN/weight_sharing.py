#!/usr/bin/python

from operator import add, sub, floordiv

from theanets import feedforward
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy as np
import theano
import theano.tensor as TT

# A class for representing a neural network with weight sharing.
class Cnn(feedforward.Classifier):
  def __init__(self, img_shape, filter_shapes, pool_sizes = "default", *args, **kwargs):
    self.img_shape = img_shape
    self.conv_weights = []
    self.conv_biases = []
    self.filter_shapes = []
    self.pool_sizes = []
    self.pool_mult_biases = []
    self.pool_add_biases = []

    if pool_sizes != "default":
      for shape, size in zip(filter_shapes, pool_sizes):
        self._add_layer_pair(shape, size)
    else:
      for shape in filter_shapes:
        self._add_layer_pair(shape, (2, 2))

    super(Cnn, self).__init__(*args, no_x = True, **kwargs)

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
    rng = np.random.RandomState(23455)
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

  # Builds a graph for the CNN.
  def _make_graph(self):
    self._inputs = TT.dtensor4("inputs")
    layer_outputs = self._inputs
    next_shape = list(self.img_shape)
    for i in range(0, len(self.conv_weights)):
      # Perform the convolution.
      filter_shape = list(self.filter_shapes[i])
      conv_out = conv.conv2d(layer_outputs, self.conv_weights[i],
          filter_shape = filter_shape,
          image_shape = next_shape)
      
      # Keep track of the shape of our output.
      next_shape[2:] = map(sub, next_shape[2:], filter_shape[2:])
      next_shape[2:] = map(add, next_shape[2:], [1, 1])
      next_shape[1] = filter_shape[1]

      # Downsample the feature maps.
      pooled_out = downsample.max_pool_2d(conv_out, self.pool_sizes[i],
          ignore_border = True)
      
      next_shape[2:] = map(floordiv, next_shape[2:], self.pool_sizes[i])

      # Account for the bias. Since it is a vector, we first need to reshape it
      # to (1, n_filters, 1, 1).
      # TODO (danielp): Have it use the specified activation.
      layer_outputs = TT.nnet.sigmoid(pooled_out + \
          self.conv_biases[i].dimshuffle("x", 0, "x", "x"))

    # Concatenate output maps into one long vector and set it as the input for
    # the normal part of our network.
    self.x = TT.flatten(layer_outputs)

  def _compile(self):
    #self._compute = theano.function([self._inputs], self.hiddens + [self.y])
    self._compute = theano.function([self._inputs], self.x.shape)

  def predict(self, inputs):
    self._compile()
    return self._compute(inputs)

  @property
  def params(self):
    params = super(feedforward.Classifier, self).params
    params.extend(self.conv_weights)
    params.extend(self.conv_biases)
