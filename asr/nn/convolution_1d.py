from chainer.links.connection.convolution_nd import ConvolutionND
from chainer.functions.connection import convolution_nd
from chainer.utils import conv_nd
from chainer import initializers
from chainer import variable

class Convolution1D(ConvolutionND):
	def __init__(self, in_channels, out_channels, nobias=False, initialW=None, initial_bias=None, cover_all=False):
		super(ConvolutionND, self).__init__()
		self.out_channels = out_channels
		self.stride = 1
		self.pad = 0
		self.cover_all = cover_all
		self.initialW = initialW
		self.ksize = conv_nd.as_tuple(1, 1)

		with self.init_scope():
			self.W = variable.Parameter(initializers._get_initializer(initialW))
			if in_channels is not None:
				self._initialize_params(in_channels)

			if nobias:
				self.b = None
			else:
				if initial_bias is None:
					initial_bias = 0
				initial_bias = initializers._get_initializer(initial_bias)
				self.b = variable.Parameter(initial_bias, out_channels)

	def _initialize_params(self, in_channels):
		self.in_channels = in_channels
		self.W.initialize((self.out_channels, self.in_channels) + self.ksize)

	def __call__(self, x):
		if self.W.data is None:
			self._initialize_params(x.shape[1])

		return convolution_nd.convolution_nd(x, self.W, self.b, self.stride, self.pad, cover_all=self.cover_all)