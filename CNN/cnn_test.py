#!/usr/bin/python

from PIL import Image
from weight_sharing import Cnn

import numpy
import pylab

img = Image.open(open("tux.jpg"))
img = numpy.asarray(img, dtype = "float64") / 256
width = img.shape[1]
height = img.shape[0]
print "Width: %d, Height: %d" % (width, height)

# Put image in 4d tensor.
img = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, width, height)
print img.shape

cnn = Cnn(img.shape, [(3, 3, 9, 9), (3, 3, 9, 9)], layers = (10, 10),
    activation = "sigmoid")
print cnn.predict(img)
