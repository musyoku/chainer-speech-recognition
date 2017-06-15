import collections
import numpy
import six
import math
import chainer
from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer import variable


def _logsumexp(a, xp, axis=None):
	vmax = xp.amax(a, axis=axis, keepdims=True)
	vmax += xp.log(xp.sum(xp.exp(a - vmax), axis=axis, keepdims=True, dtype=a.dtype))
	return xp.squeeze(vmax, axis=axis)

def _softmax(x, xp):
	val = xp.exp(x - xp.amax(x, axis=2, keepdims=True))
	val /= xp.sum(val, axis=2, keepdims=True)
	return val + 1e-8

def _label_to_path(labels, blank_symbol, xp):
	path = xp.full((len(labels), labels.shape[1] * 2 + 1),
				   blank_symbol, dtype=numpy.int32)
	path[:, 1::2] = labels
	return path


def _log_dot(prob, rr, xp):
	return _logsumexp(prob + xp.swapaxes(rr, 1, 2), xp, axis=2)


def _move_label_to_back(path, path_length, xp):
	s1 = path.shape[1]
	index = (xp.arange(0, path.size, s1, dtype=numpy.int32)[:, None] + (xp.arange(s1) + path_length[:, None])[:, ::-1] % s1)
	return xp.take(path, index)


def _move_inputs(prob, input_length, xp):
	seq_length, batch_size, num_labels = prob.shape
	rotate = (xp.arange(seq_length)[:, None] + input_length) % seq_length
	index = rotate * batch_size + xp.arange(batch_size)
	return xp.take(prob.reshape(seq_length * batch_size, num_labels), index, axis=0)


class ConnectionistTemporalClassification(function.Function):

	"""The implementation of Connectionist Temporal Classfication loss functions.

	To make it usable for real-world cases, this class has two policies below.
	1. This class computes forward and backward variables in the log domain.
	2. This class applies the softmax function to inputs. The Backward
	values of CTC loss is often overflows. This is avoided by computing
	backward values before the activation function is applied.
	"""

	def __init__(self, blank_symbol, reduce='mean'):
		self.blank_symbol = blank_symbol
		self.zero_padding = -10000000000.0
		# self.zero_padding = -1000.0

		if reduce not in ('mean', 'no'):
			raise ValueError(
				"only 'mean' and 'no' are valid "
				"for 'reduce', but '%s' is given" % reduce)
		self.reduce = reduce

	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() > 3)  # TODO(okuta): > 3?
		l_type = in_types[2]
		type_check.expect(l_type.dtype == numpy.int32)

		x_basetype = in_types[3]  # TODO(oktua): Check x_basetype size

		for i in six.moves.range(3, len(in_types)):
			x_type = in_types[i]
			type_check.expect(
				x_type.dtype == numpy.float32,
				x_type.shape == x_basetype.shape,
			)

	def log_matrix(self, x, xp, fill=True):
		if xp == numpy:
			res = numpy.ma.log(x)
			if fill:
				res = res.filled(fill_value=self.zero_padding)
		else:
			if fill:
				create_recurrence_relation = cuda.cupy.ElementwiseKernel(
					'T x, T e', 'T y',
					'y = x == 0 ? e : log(x)',
					'create_recurrence_relation')
				res = create_recurrence_relation(x, self.zero_padding)
			else:
				res = xp.log(x)
		return res.astype(numpy.float32)

	def recurrence_relation(self, label, path_length, max_path_length, dtype, xp):
		"""Transition in forword and backword algorithms is represented as matrix.

		See also
		https://blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
		"""
		# print("recurrence_relation")
		batch_size, lab = label.shape
		# print("label:")
		# print(label)
		repeat_mask = xp.ones((batch_size, lab * 2 + 1))
		# repeat_mask[:, 1::2] = label != xp.roll(label, 1, axis=1) # valid when s > 1
		repeat_mask[:, 1::2] = (label !=
								xp.take(label, xp.arange(-1, lab - 1)
										% lab + xp.arange(0, batch_size * lab,
														  lab)[:, None]))
		# print("flag:")
		# print((label !=
		# 						xp.take(label, xp.arange(-1, lab - 1)
		# 								% lab + xp.arange(0, batch_size * lab,
		# 												  lab)[:, None])))
		# print(repeat_mask)
		repeat_mask[:, 1] = 1	# correct the result of xp.roll for s = 1
		# print(repeat_mask)
		# print("eye:")
		# print(xp.eye(max_path_length, dtype=dtype))
		# print("eye k=1:")
		# print(xp.eye(max_path_length, k=1, dtype=dtype))
		# print("eye k=2:")
		# print(xp.eye(max_path_length, k=2, dtype=dtype))
		# print("a:")
		# print(xp.eye(max_path_length, k=2, dtype=dtype) * repeat_mask[:, None])
		# print("b:")
		# print(xp.eye(max_path_length, k=2, dtype=dtype) * (xp.arange(max_path_length, dtype=dtype) % dtype(2))[None, :] * repeat_mask[:, None])
		relation = (xp.eye(max_path_length, dtype=dtype)[None, :] +
			  xp.eye(max_path_length, k=1, dtype=dtype)[None, :] +
			  (xp.eye(max_path_length, k=2, dtype=dtype) *
			   (xp.arange(max_path_length, dtype=dtype) % dtype(2))[None, :] * repeat_mask[:, None]))
		# print("relation:")
		# print(relation)
		mask = (path_length[:, None] > xp.arange(max_path_length))[..., None]
		relation = relation * mask 						# remove unnecessary path
		# relation = relation * xp.swapaxes(mask, 1, 2) 	# remove unnecessary path
		# print("relation:")
		# print(relation)
		# raise Exception()
		# print("mask:")
		# print((path_length[:, None] > xp.arange(max_path_length))[..., None])
		return self.log_matrix(relation, xp)

	# path probablity to label probability
	def label_probability(self, label_size, path, path_length,
						  multiply_seq, xp):
		labels_prob = self.log_matrix(xp.zeros((len(path), label_size), dtype=multiply_seq.dtype), xp)
		ret = xp.empty((len(multiply_seq),) + labels_prob.shape, dtype=labels_prob.dtype)
		ret[...] = labels_prob
		if xp == numpy:
			for b in six.moves.range(len(path)):
				target_path = path[b][0:path_length[b]]
				chars = {c for c in target_path}
				for c in chars:
					ret[:, b, c] = _logsumexp(
						multiply_seq[:, b, 0:path_length[b]]
						[:, target_path == c], numpy, axis=1)
		else:

			# _ret = ret.copy()
			# for b in six.moves.range(len(path)):
			# 	print("multiply_seq:")
			# 	print(multiply_seq)
			# 	target_path = path[b][0:path_length[b]]
			# 	print("target_path:")
			# 	print(target_path)
			# 	chars = {int(c) for c in target_path}
			# 	for c in chars:
			# 		mask = cuda.to_cpu(target_path) == c
			# 		print("seq:")
			# 		print(multiply_seq[:, b, 0:path_length[b]][:, mask])
			# 		_ret[:, b, c] = _logsumexp(
			# 			multiply_seq[:, b, 0:path_length[b]][:, mask], xp, axis=1)

			# print("_ret:")
			# print(_ret)

			for t, multiply in enumerate(multiply_seq):
				cuda.cupy.ElementwiseKernel(
					'raw T x, raw I y, raw I path_length, I max_label_length, I max_path_length',
					'T z',
					'''
					T value = z;
					I label_idx = i % max_label_length, batch_idx = i / max_label_length;
					int ind[2] = {batch_idx, -1};
					for (int path_idx = 0; path_idx < max_path_length; ++path_idx) {
						ind[1] = path_idx;
						if (path_idx < path_length[batch_idx] && y[ind] == label_idx) {
							T xvalue = x[ind];
							T at = xvalue, bt = value;
							if (value > xvalue) {
								at = value;
								bt = xvalue;
							}
							value = at + log(1 + exp(bt - at));
						}
					}
					z = value;
					''',
					'reduce_probability')(multiply, path, path_length,
										  labels_prob.shape[1],
										  path.shape[1], ret[t])


		return ret

	def calc_trans(self, yseq, input_length, label, label_length, path, path_length, xp):
		forward_prob = self.log_matrix(xp.eye(path.shape[1], dtype='f')[0], xp)[None, :]
		backward_prob = forward_prob
		offset = xp.arange(0, yseq[0].size, yseq[0].shape[1], dtype=path.dtype)[:, None]

		# print("yseq:")
		# print(yseq)
		# prob[i] := forward[i] + backward[-i-1]
		index = offset + path
		forward_relation = self.recurrence_relation(label, path_length, path.shape[1], numpy.float32, xp)
		prob = xp.empty((len(yseq),) + index.shape, dtype=forward_prob.dtype)
		# forward computation.
		for i, y in enumerate(yseq):
			# calc forward probability in log scale
			# print("take:")
			take = xp.take(y, index)
			# print(take)
			forward_prob = xp.take(y, index) + _log_dot(forward_prob[:, None, :], forward_relation, xp)
			prob[i] = forward_prob
			# print("prob[{}]:".format(i))
			# print(prob[i])
			# print("forward_prob:")
			# print(forward_prob)
		r_index = offset + _move_label_to_back(path, path_length, xp)

		# print("label:")
		# print(label)
		# print("moved label:")
		# print(_move_label_to_back(label, label_length, xp))
		# print("input_length:")
		# print(input_length)

		# rotate yseq with path_length
		yseq_inv = _move_inputs(yseq, input_length, xp)[::-1]
		backward_relation = self.recurrence_relation(_move_label_to_back(label, label_length, xp), path_length, path.shape[1], numpy.float32, xp)
		# print("backward_relation:")
		# print(backward_relation)
		# print("yseq:")
		# print(yseq)
		# print(input_length)
		# print("yseq_inv:")
		# print(yseq_inv)
		# move to back.
		# print("prob:")
		# print(prob)
		prob = _move_inputs(prob, input_length, xp)
		# print("prob:")
		# print(prob)
		# print("backward_prob:")
		# print(backward_prob)

		# backward computation.
		ps1 = path.shape[1]
		backward_prob_index = (
			xp.arange(0, path.size, ps1, dtype=numpy.int32)[:, None] +
			(xp.arange(ps1) - path_length[:, None]) % ps1)
		for i, y_inv in enumerate(yseq_inv):
			# calc backward probability
			backward_prob = _log_dot(backward_prob[:, None, :], backward_relation, xp)
			_prob = xp.take(backward_prob[:, ::-1], backward_prob_index)
			# print("backward_prob:")
			# print(backward_prob)
			# print("_prob:")
			# print(_prob)
			# print("prob[{}] before:".format(-i-1))
			# print(prob[-i-1])
			prob[-i - 1] += _prob
			# print("prob[{}]:".format(-i-1))
			# print(prob[-i-1])
			backward_prob = xp.take(y_inv, r_index) + backward_prob
			# print("take:")
			# print(xp.take(y_inv, r_index))

		# print("prob:")
		# print(prob)

		# move to front.
		return _move_inputs(prob, -self.input_length, xp)

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		self.input_length = inputs[0]
		label_length = inputs[1]
		t = inputs[2]
		xs = inputs[3:]

		if chainer.is_debug():
			# Batch size check.
			assert len(xs[0]) == len(t)
			assert len(xs[0]) == len(self.input_length)
			assert len(xs[0]) == len(label_length)

			# Length check.
			assert len(xs) >= xp.max(self.input_length)
			assert len(t[0]) >= xp.max(label_length)

		self.path_length = 2 * label_length + 1

		yseq_shape = (len(xs),) + xs[0].shape
		self.yseq = _softmax(xp.vstack(xs).reshape(yseq_shape), xp)
		# print("yseq softmax:")
		# print(self.yseq)
		# print(self.yseq.shape)
		log_yseq = self.log_matrix(self.yseq, xp, fill=False)
		self.path = _label_to_path(t, self.blank_symbol, xp)
		# print(self.path)
		self.prob_trans = self.calc_trans(
			log_yseq, self.input_length, t,
			label_length, self.path, self.path_length, xp)

		# print("prob_trans:")
		# print(self.prob_trans)
		# print(self.prob_trans.shape)
		# print(self.prob_trans[0])

		loss = -_logsumexp(self.prob_trans[0], xp, axis=1)
		if self.reduce == 'mean':
			loss = utils.force_array(xp.mean(loss))
		return loss,

	def backward(self, inputs, grad_output):
		xp = cuda.get_array_module(inputs[0])
		batch_size = len(inputs[2])

		# print(inputs[0])
		# print(inputs[1])
		# print(inputs[2])

		total_probability = _logsumexp(self.prob_trans[0], xp, axis=1)
		# print("total_probability:")
		# print(total_probability.shape)
		# print(total_probability)

		mask = xp.arange(len(self.yseq))[:, None] < self.input_length
		# print(mask.shape)
		# print(mask)


		if chainer.is_debug():
			_total_probability = _logsumexp(self.prob_trans, xp, axis=2) * mask
			__total_probability = _total_probability + total_probability * (1 - mask)

			# total probability should be constant regardless of t
			std = xp.std(__total_probability, axis=0)
			threshold = 1e-3
			
			# print(inputs[0])
			# print(inputs[1])
			# print(inputs[2])
			# print("_total_probability:")
			# print(_total_probability.shape)
			# print(_total_probability)
			# print("__total_probability:")
			# print(__total_probability.shape)
			# print(__total_probability)
			# print("std:")
			# print(std)

			if len(std[std > threshold]) != 0:
				print(inputs[0])
				print(inputs[1])
				print(inputs[2])
				print("_total_probability:")
				print(_total_probability.shape)
				print(_total_probability)
				print("__total_probability:")
				print(__total_probability.shape)
				print(__total_probability)
				print("std:")
				print(std)

			assert len(std[std > threshold]) == 0

			minimum_log_prob = self.zero_padding / 1e2

			
			if len(__total_probability[__total_probability < minimum_log_prob]) != 0:
				print(inputs[0])
				print(inputs[1])
				print(inputs[2])
				print("_total_probability:")
				print(_total_probability.shape)
				print(_total_probability)
				print("__total_probability:")
				print(__total_probability.shape)
				print(__total_probability)
				print("std:")
				print(std)

			assert len(__total_probability[__total_probability < minimum_log_prob]) == 0

		label_prob = self.label_probability(
			self.yseq.shape[2], self.path, self.path_length,
			self.prob_trans, xp)
		# print("label_prob:")
		# print(label_prob.shape)
		# print(label_prob)

		self.yseq -= xp.exp(label_prob - total_probability[:, None])
		if self.reduce == 'mean':
			self.yseq *= grad_output[0] / batch_size
		else:
			self.yseq *= grad_output[0][..., None]
		# mask
		self.yseq *= (
			xp.arange(len(self.yseq))[:, None] < self.input_length)[..., None]
		return (None, None, None) + tuple([y for y in self.yseq])


def connectionist_temporal_classification(
		x, t, blank_symbol, input_length=None, label_length=None,
		reduce='mean'):
	"""Connectionist Temporal Classification loss function.

	Connectionist Temporal Classification(CTC) [Graves2006]_ is a loss function
	of sequence labeling where the alignment between the inputs and target is
	unknown. See also [Graves2012]_

	The output is a varialbe whose value depends on the value of
	the option ``reduce``. If it is ``'no'``, it holds the samplewise
	loss values. If it is ``'mean'``, it takes the mean of loss values.


	Args:
		x (sequence of Variable): RNN output at each time. ``x`` must be a list
			of :class:`~chainer.Variable` s. Each element of ``x``, ``x[i]``
			is a :class:`~chainer.Variable` representing output of RNN at time
			``i``.
		t (Variable): Expected label sequence.
		blank_symbol (int): Index of blank_symbol.
			This value must be non-negative.
		input_length (Variable): Length of valid sequence for each of mini
			batch ``x`` (optional). If input_length is skipped, It regards that
			all of ``x`` is valid input.
		label_length (Variable): Length of valid sequence for each of mini
			batch ``t`` (optional). If label_length is skipped, It regards that
			all of ``t`` is valid input.
		reduce (str): Reduction option. Its value must be either
			``'mean'`` or ``'no'``. Otherwise,
			:class:`ValueError` is raised.

	Returns:
	   ~chainer.Variable:
		   A variable holding a scalar value of the CTC loss.
		   If ``reduce`` is ``'no'``, the output varialbe holds array
		   whose shape is `(B,)` where `B` is the number of samples.
		   If it is ``'mean'``, it holds a scalar.

	.. note::
	   You need to input ``x`` without applying to activation functions(e.g.
	   softmax function), because this function applies softmax functions
	   to ``x`` before calculating CTC loss to avoid numerical limitations.
	   You also need to apply softmax function to forwarded values before you
	   decode it.

	.. note::
	   This function is differentiable only by ``x``.

	.. note::
	   This function supports (batch, sequence, 1-dimensional input)-data.

	.. [Graves2006] Alex Graves, Santiago Fernandez,\
	Faustino Gomez, Jurgen Schmidhuber,\
	`Connectionist Temporal Classification: Labelling Unsegmented\
	Sequence Data with Recurrent Neural Networks\
	<ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf>`_

	.. [Graves2012] Alex Graves,\
	`Supervised Sequence Labelling with Recurrent Neural Networks\
	<http://www.cs.toronto.edu/~graves/preprint.pdf>`_

	"""
	if not isinstance(x, collections.Sequence):
		raise TypeError('x must be a list of Variables')
	if not isinstance(blank_symbol, int):
		raise TypeError('blank_symbol must be non-negative integer.')
	assert blank_symbol >= 0
	assert blank_symbol < x[0].shape[1]
	# This implementation only supports 1-dimensional data.
	# TODO(jnishi): Support d(>1)-dimentinal inputs.
	assert(len(x[0].shape) == 2)

	if input_length is None:
		xp = cuda.get_array_module(x[0].data)
		input_length = variable.Variable(
			xp.full((len(x[0].data),), len(x), dtype=numpy.int32))
		label_length = variable.Variable(
			xp.full((len(t.data),), len(t.data[0]), dtype=numpy.int32))

	return ConnectionistTemporalClassification(blank_symbol, reduce)(
		input_length, label_length, t, *x)
