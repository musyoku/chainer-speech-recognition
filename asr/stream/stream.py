import chainer, math
from chainer import functions, cuda, links, variable
from chainer.links import *
from .convolution_2d import Convolution2D as WeightnormConvolution2D
from .layernorm import normalize_layer

# Standar functions

class ClippedReLU():
	def __init__(self, z=20):
		self.z = z

	def __call__(self, x):
		return functions.clipped_relu(x, self.z)

class CReLU():
	def __init__(self, axis=1):
		self.axis = axis

	def __call__(self, x):
		return functions.crelu(x, self.axis)

class ELU():
	def __init__(self, alpha=1):
		self.alpha = alpha

	def __call__(self, x):
		return functions.elu(x, self.alpha)
	
def HardSigmoid():
	return functions.hard_sigmoid

class LeakyReLU():
	def __init__(self, slope=1):
		self.slope = slope

	def __call__(self, x):
		return functions.leaky_relu(x, self.slope)
	
def LogSoftmax():
	return functions.log_softmax

class Maxout():
	def __init__(self, pool_size=0.5):
		self.pool_size = pool_size

	def __call__(self, x):
		return functions.maxout(x, self.pool_size)
	
def ReLU():
	return functions.relu

def Sigmoid():
	return functions.sigmoid

class Softmax():
	def __init__(self, axis=1):
		self.axis = axis

	def __call__(self, x):
		return functions.softmax(x, self.axis)

class Softplus():
	def __init__(self, beta=1):
		self.beta = beta

	def __call__(self, x):
		return functions.softplus(x, self.beta)

def Tanh():
	return functions.tanh

# Pooling

class AveragePooling2D():
	def __init__(self, ksize, stride=None, pad=0):
		self.ksize = ksize
		self.stride = stride
		self.pad = pad

	def __call__(self, x):
		return functions.average_pooling_2d(x, self.ksize, self.stride, self.pad)

class AveragePoolingND():
	def __init__(self, ksize, stride=None, pad=0):
		self.ksize = ksize
		self.stride = stride
		self.pad = pad

	def __call__(self, x):
		return functions.average_pooling_nd(x, self.ksize, self.stride, self.pad)

class MaxPooling2D():
	def __init__(self, ksize, stride=None, pad=0, cover_all=True):
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.cover_all = cover_all

	def __call__(self, x):
		return functions.max_pooling_2d(x, self.ksize, self.stride, self.pad)

class MaxPoolingND():
	def __init__(self, ksize, stride=None, pad=0, cover_all=True):
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.cover_all = cover_all

	def __call__(self, x):
		return functions.max_pooling_nd(x, self.ksize, self.stride, self.pad)

class SpatialPyramidPooling2D():
	def __init__(self, pyramid_height, pooling_class):
		self.pyramid_height = pyramid_height
		self.pooling_class = pooling_class

	def __call__(self, x):
		return functions.spatial_pyramid_pooling_2d(x, self.pyramid_height, self.pooling_class)

class Unpooling2D():
	def __init__(self, ksize, stride=None, pad=0, outsize=None, cover_all=True):
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.outsize = outsize
		self.cover_all = cover_all

	def __call__(self, x):
		return functions.unpooling_2d(x, self.ksize, self.stride, self.pad, self.outsize, self.cover_all)

class UpSampling2D():
	def __init__(self, indexes, ksize, stride=None, pad=0, outsize=None, cover_all=True):
		self.indexes = indexes
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.outsize = outsize
		self.cover_all = cover_all

	def __call__(self, x):
		return functions.upsampling_2d(x, self.indexes, self.ksize, self.stride, self.pad, self.outsize, self.cover_all)

# Array manipulations

class BroadcastTo():
	def __init__(self, shape):
		self.shape = shape

	def __call__(self, x):
		return functions.broadcast_to(x, self.shape)

class ExpandDims():
	def __init__(self, axis):
		self.axis = axis

	def __call__(self, x):
		return functions.expand_dims(x, self.axis)

def Flatten():
	return functions.flatten

class Reshape():
	def __init__(self, shape):
		self.shape = shape

	def __call__(self, x):
		return functions.reshape(x, self.shape)

class RollAxis():
	def __init__(self, axis, start=0):
		self.axis = axis
		self.start = start

	def __call__(self, x):
		return functions.rollaxis(x, self.axis, self.start)

class Squeeze():
	def __init__(self, axis):
		self.axis = axis

	def __call__(self, x):
		return functions.squeeze(x, self.axis)

class SwapAxes():
	def __init__(self, axis1, axis2):
		self.axis1 = axis1
		self.axis2 = axis2

	def __call__(self, x):
		return functions.swapaxes(x, self.axis1, self.axis2)

class Tile():
	def __init__(self, reps):
		self.reps = reps

	def __call__(self, x):
		return functions.tile(x, self.reps)

class Transpose():
	def __init__(self, axes):
		self.axes = axes

	def __call__(self, x):
		return functions.transpose(x, self.axes)

# Noise injections

class Dropout():
	def __init__(self, ratio=0.5):
		self.ratio = ratio

	def __call__(self, x):
		if self.ratio == 0:
			return x
		return functions.dropout(x, self.ratio)

class GaussianNoise():
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, x):
		if chainer.config.train == False:
			return x
		xp = cuda.get_array_module(x.data)
		std = math.log(self.std ** 2)
		noise = functions.gaussian(chainer.Variable(xp.zeros_like(x.data)), chainer.Variable(xp.full_like(x.data, std)))
		return x + noise

# Link

def Convolution2D(in_channel, out_channel, ksize, stride=1, pad=0, initialW=None, weightnorm=False):
	if weightnorm:
		return WeightnormConvolution2D(in_channel, out_channel, ksize, stride=1, pad=pad, initialV=initialW)
	return links.Convolution2D(in_channel, out_channel, ksize, stride=1, pad=pad, initialW=initialW)

class LayerNormalization(chainer.link.Link):
	def __init__(self, size=None, eps=1e-6, initial_gamma=None, initial_beta=None):
		super(LayerNormalization, self).__init__()
		if initial_gamma is None:
			initial_gamma = 1
		if initial_beta is None:
			initial_beta = 0

		with self.init_scope():
			self.gamma = variable.Parameter(initial_gamma)
			self.beta = variable.Parameter(initial_beta)
			self.eps = eps

		if size is not None:
			self._initialize_params(size)

	def _initialize_params(self, size):
		self.gamma.initialize(size)
		self.beta.initialize(size)

	def __call__(self, x):
		if self.gamma.data is None:
			self._initialize_params(x.shape[1])

		normalized = normalize_layer(x)
		return functions.math.bias.bias(functions.math.scale.scale(normalized, self.gamma), self.beta)

class GLU(object):
	def __init__(self, in_channels, out_channels, ksize=(3, 5), pad=0, wgain=1., weightnorm=False):
		wstd = math.sqrt(wgain / in_channels / ksize[0] / ksize[1])
		self.W = Convolution2D(in_channels, 2 * out_channels, ksize, stride=1, pad=pad, initialW=chainer.initializers.HeNormal(wstd), weightnorm=weightnorm)
		self._in_channels, self._out_channels, self._kernel_size, = in_channels, out_channels, ksize

	def __call__(self, X):
		pad = self._kernel_size[1] - 1
		WX = self.W(X)
		if pad > 0:
			WX = WX[..., :-pad]

		A, B = functions.split_axis(WX, 2, axis=1)
		H = A * functions.sigmoid(B)
		return H

# Connections

class Residual(object):
	def __init__(self, *layers):
		self.layers = layers

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

# Chain

class Stream(chainer.Chain):
	def __init__(self, *layers):
		super(Stream, self).__init__()
		assert not hasattr(self, "layers")
		self.layers = []
		if len(layers) > 0:
			self.layer(*layers)

	def layer(self, *layers):
		with self.init_scope():
			for i, layer in enumerate(layers):
				index = i + len(self.layers)

				if isinstance(layer, chainer.Link):
					setattr(self, "layer_%d" % index, layer)

				if isinstance(layer, GLU):
					setattr(self, "layer_%d" % index, layer.W)

				if isinstance(layer, Residual):
					for _index, _layer in enumerate(layer.layers):
						if isinstance(_layer, chainer.Link):
							setattr(self, "layer_{}_{}".format(index, _index), _layer)
							
		self.layers += layers

	def __call__(self, x):
		for layer in self.layers:
			y = layer(x)
			if isinstance(layer, Residual):
				y += x
			x = y
		return x