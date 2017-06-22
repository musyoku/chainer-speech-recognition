import unittest
import mock
import numpy
import six
import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.normalization import batch_normalization
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from layernorm import normalize_layer, NormalizeLayer

def _batch_normalization(expander, gamma, beta, x, mean, var):
	mean = mean[expander]
	std = numpy.sqrt(var)[expander]
	y_expect = (gamma[expander] * (x - mean) / std + beta[expander])
	return y_expect

def _normalize_layer(x, eps):
	size = x.shape[1] * x.shape[2]
	mean = functions.math.sum.sum(x, axis=(1, 2), keepdims=True) / size
	mean = functions.array.broadcast.broadcast_to(mean, x.shape)
	std = functions.math.sum.sum(functions.math.square.square(x - mean), axis=(1, 2), keepdims=True) / size
	std = functions.math.sqrt.sqrt(std) + eps
	std = functions.array.broadcast.broadcast_to(std, x.shape)
	return (x - mean) / std

@testing.parameterize(*(testing.product({
	'x_shape': [(3, 40, 10)],
	'y_shape': [(3, 40, 10)],
	# 'y_shape': [(1, 1, 10)],
	'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestLayerNormalization(unittest.TestCase):

	def setUp(self):
		self.eps = 1e-6
		x_shape = (5,) + self.x_shape
		y_shape = (5,) + self.y_shape
		self.x = numpy.random.uniform(-10, 10, x_shape).astype(self.dtype)
		self.gy = numpy.random.uniform(-1, 1, y_shape).astype(self.dtype)
		self.train = True
		self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-3}
		self.check_backward_options = {'dtype': numpy.float64}
		if self.dtype == numpy.float16:
			self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
			self.check_backward_options = {
				'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}

	def check_forward(self, x, use_cudnn='always'):
		with chainer.using_config('use_cudnn', use_cudnn):
			y = normalize_layer(x, eps=self.eps)
		self.assertEqual(y.data.dtype, self.dtype)

		y_expect = _normalize_layer(self.x, self.eps).data

		testing.assert_allclose(y_expect, y.data, **self.check_forward_options)

	@condition.retry(3)
	def test_forward_cpu(self):
		self.check_forward(self.x)

	@attr.gpu
	@condition.retry(3)
	def test_forward_gpu(self):
		self.check_forward(self.x)

	def check_backward(self, x, y_grad, use_cudnn='always'):
		with chainer.using_config('use_cudnn', use_cudnn), chainer.using_config('train', self.train):
			gradient_check.check_backward(
				NormalizeLayer(self.eps), x, y_grad,
				**self.check_backward_options)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.x, self.gy)

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu(self):
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu_no_cudnn(self):
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), 'never')

testing.run_module(__name__, __file__)