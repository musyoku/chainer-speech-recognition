from __future__ import division
from __future__ import print_function
from six.moves import xrange
import math
from chainer import function, cuda
from chainer.functions.array.broadcast import _backward_one

def _backward_sum(gy, in_shape):
	xp = cuda.get_array_module(gy)
	sum_axis = (1, 2)
	keepdims = True
	if not (len(in_shape) == 0 or sum_axis is None or keepdims):
		actual_axis = []
		for axis in sum_axis:
			if axis < 0:
				axis += len(in_shape)
			actual_axis.append(axis)
		for axis in sorted(actual_axis):
			gy = xp.expand_dims(gy, axis=axis)
	if hasattr(xp, 'broadcast_to'):
		gx = xp.broadcast_to(gy, in_shape)
	else:
		# NumPy 1.9 does not support broadcast_to.
		dummy_x = xp.empty(in_shape, 'b')
		gx, _ = xp.broadcast_arrays(gy, dummy_x)

	return gx

class NormalizeLayer(function.Function):
	def __init__(self, eps=2e-5):
		self.eps = eps

	def forward(self, xs, eps=1e-6):
		self.retain_inputs(())
		self.eps = eps
		x = xs[0]
		self.x_shape = x.shape
		self.x_dtype = x.dtype
		xp = cuda.get_array_module(x)
		size = x.shape[1] * x.shape[2]
		self.x_size = size
		mean = xp.mean(x, axis=(1, 2), keepdims=True)
		self.broadcast_shape = mean.shape
		self.diff = x - mean
		std = xp.std(x, axis=(1, 2), keepdims=True)
		self.std = std
		return self.diff / std,

	def backward(self, xs, grads):
		xp = cuda.get_array_module(grads[0])

		std_grad = _backward_one(xp, self.broadcast_shape, self.x_dtype, -grads[0] / (self.std ** 2) * self.diff)
		var_grad = _backward_sum(std_grad * 0.5 / self.std, self.x_shape) / self.x_size
		var_grad = var_grad * 2 * self.diff
		var_grad += grads[0] / self.std

		grad_broad = _backward_one(xp, self.broadcast_shape, self.x_dtype, -var_grad)
		mean_grad = _backward_sum(grad_broad, self.x_shape) / self.x_size

		return var_grad + mean_grad,

def normalize_layer(x, eps=1e-6):
	return NormalizeLayer(eps)(x)

