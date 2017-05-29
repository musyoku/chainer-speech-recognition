from __future__ import division
from __future__ import print_function
from six.moves import xrange
import math
import numpy as np
import chainer
from chainer import cuda, Variable, function, link, functions, links, initializers
from chainer.utils import type_check
from chainer.links import EmbedID, Linear, BatchNormalization, Convolution2D
from convolution_2d import Convolution2D as WeightnormConvolution2D

class GLU(link.Chain):
	def __init__(self, in_channels, out_channels, kernel_size=(3, 5), wgain=1., weightnorm=False):
		wstd = math.sqrt(wgain / in_channels / kernel_size[0] / kernel_size[1])
		if weightnorm:
			W = WeightnormConvolution2D(in_channels, 2 * out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialV=initializers.HeNormal(wstd))
		else:
			W = Convolution2D(in_channels, 2 * out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.HeNormal(wstd))

		super(GLU, self).__init__(W=W)
		self._in_channels, self._out_channels, self._kernel_size, = in_channels, out_channels, kernel_size
		self.reset_state()

	def __call__(self, X):
		# remove right paddings
		# e.g.
		# kernel_size = 3
		# pad = 2
		# input sequence with paddings:
		# [0, 0, x1, x2, x3, 0, 0]
		# |< t1 >|
		#     |< t2 >|
		#         |< t3 >|
		pad = self._kernel_size[1] - 1
		WX = self.W(X)[..., :-pad]

		A, B = functions.split_axis(WX, 2, axis=1)
		self.H = A * functions.sigmoid(B)
		return self.H

	def forward_one_step(self, X):
		pad = self._kernel_size[1] - 1
		wx = self.W(X)[:, :, -pad-1, None]
		a, b = functions.split_axis(wx, 2, axis=1)
		h = a * functions.sigmoid(b)

		if self.H is None:
			self.H = h
		else:
			self.H = functions.concat((self.H, h), axis=2)

		return self.H

	def reset_state(self):
		self.set_state(None)

	def set_state(self, H):
		self.H = H

	def get_all_hidden_states(self):
		return self.H

class QRNNEncoder(GLU):
	pass

class QRNNDecoder(GLU):
	def __init__(self, in_channels, out_channels, kernel_size=2, pooling="f", zoneout=False, zoneout_ratio=0.1, wgain=1.):
		super(QRNNDecoder, self).__init__(in_channels, out_channels, kernel_size, pooling, zoneout, zoneout_ratio, wgain=wgain)
		self.num_split = len(pooling) + 1
		wstd = math.sqrt(wgain / in_channels / kernel_size)
		self.add_link("V", links.Linear(out_channels, self.num_split * out_channels, initialW=initializers.Normal(wstd)))

	# ht_enc is the last encoder state
	def __call__(self, X, ht_enc):
		pad = self._kernel_size - 1
		WX = self.W(X)
		if pad > 0:
			WX = WX[:, :, :-pad]
		Vh = self.V(ht_enc)

		# copy Vh
		# e.g.
		# WX = [[[  0	1	2]
		# 		 [	3	4	5]
		# 		 [	6	7	8]
		# Vh = [[11, 12, 13]]
		# 
		# Vh, WX = F.broadcast(F.expand_dims(Vh, axis=2), WX)
		# 
		# WX = [[[  0	1	2]
		# 		 [	3	4	5]
		# 		 [	6	7	8]
		# Vh = [[[ 	11	11	11]
		# 		 [	12	12	12]
		# 		 [	13	13	13]
		Vh, WX = functions.broadcast(functions.expand_dims(Vh, axis=2), WX)

		return self.pool(functions.split_axis(WX + Vh, self.num_split, axis=1))

	def forward_one_step(self, X, ht_enc):
		pad = self._kernel_size - 1
		WX = self.W(X)[:, :, -pad-1, None]
		Vh = self.V(ht_enc)

		Vh, WX = functions.broadcast(functions.expand_dims(Vh, axis=2), WX)

		return self.pool(functions.split_axis(WX + Vh, self.num_split, axis=1))

class QRNNGlobalAttentiveDecoder(QRNNDecoder):
	def __init__(self, in_channels, out_channels, kernel_size=2, zoneout=False, zoneout_ratio=0.1, wgain=1.):
		super(QRNNGlobalAttentiveDecoder, self).__init__(in_channels, out_channels, kernel_size, "fo", zoneout, zoneout_ratio, wgain=wgain)
		wstd = math.sqrt(wgain / in_channels / kernel_size)
		self.add_link('o', links.Linear(2 * out_channels, out_channels, initialW=initializers.Normal(wstd)))

	# X is the input of the decoder
	# ht_enc is the last encoder state
	# H_enc is the encoder's las layer's hidden sates
	def __call__(self, X, ht_enc, H_enc, skip_mask=None):
		pad = self._kernel_size - 1
		WX = self.W(X)
		if pad > 0:
			WX = WX[:, :, :-pad]
		Vh = self.V(ht_enc)
		Vh, WX = functions.broadcast(functions.expand_dims(Vh, axis=2), WX)

		# f-pooling
		Z, F, O = functions.split_axis(WX + Vh, 3, axis=1)
		Z = functions.tanh(Z)
		F = self.zoneout(F)
		O = functions.sigmoid(O)
		T = Z.shape[2]

		# compute ungated hidden states
		self.contexts = []
		for t in xrange(T):
			z = Z[:, :, t]
			f = F[:, :, t]
			if t == 0:
				ct = (1 - f) * z
				self.contexts.append(ct)
			else:
				ct = f * self.contexts[-1] + (1 - f) * z
				self.contexts.append(ct)

		if skip_mask is not None:
			assert skip_mask.shape[1] == H_enc.shape[2]
			softmax_getas = (skip_mask == 0) * -1e6

		# compute attention weights (eq.8)
		H_enc = functions.swapaxes(H_enc, 1, 2)
		for t in xrange(T):
			ct = self.contexts[t]
			geta = 0 if skip_mask is None else softmax_getas[..., None]	# to skip PAD
			mask = 1 if skip_mask is None else skip_mask[..., None]		# to skip PAD
			alpha = functions.batch_matmul(H_enc, ct) + geta
			alpha = functions.softmax(alpha) * mask
			alpha = functions.broadcast_to(alpha, H_enc.shape)	# copy
			kt = functions.sum(alpha * H_enc, axis=1)
			ot = O[:, :, t]
			self.ht = ot * self.o(functions.concat((kt, ct), axis=1))

			if t == 0:
				self.H = functions.expand_dims(self.ht, 2)
			else:
				self.H = functions.concat((self.H, functions.expand_dims(self.ht, 2)), axis=2)

		return self.H

	def forward_one_step(self, X, ht_enc, H_enc, skip_mask):
		pad = self._kernel_size - 1
		WX = self.W(X)[:, :, -pad-1, None]
		Vh = self.V(ht_enc)

		Vh, WX = functions.broadcast(functions.expand_dims(Vh, axis=2), WX)

		# f-pooling
		Z, F, O = functions.split_axis(WX + Vh, 3, axis=1)
		Z = functions.tanh(Z)
		F = self.zoneout(F)
		O = functions.sigmoid(O)
		T = Z.shape[2]

		# compute ungated hidden states
		for t in xrange(T):
			z = Z[:, :, t]
			f = F[:, :, t]
			if self.contexts is None:
				ct = (1 - f) * z
				self.contexts = [ct]
			else:
				ct = f * self.contexts[-1] + (1 - f) * z
				self.contexts.append(ct)

		if skip_mask is not None:
			assert skip_mask.shape[1] == H_enc.shape[2]
			softmax_getas = (skip_mask == 0) * -1e6

		# compute attention weights (eq.8)
		H_enc = functions.swapaxes(H_enc, 1, 2)
		for t in xrange(T):
			ct = self.contexts[t - T]
			geta = 0 if skip_mask is None else softmax_getas[..., None]	# to skip PAD
			mask = 1 if skip_mask is None else skip_mask[..., None]		# to skip PAD
			alpha = functions.batch_matmul(H_enc, ct) + geta
			alpha = functions.softmax(alpha) * mask
			alpha = functions.broadcast_to(alpha, H_enc.shape)	# copy
			kt = functions.sum(alpha * H_enc, axis=1)
			ot = O[:, :, t]
			self.ht = ot * self.o(functions.concat((kt, ct), axis=1))

			if self.H is None:
				self.H = functions.expand_dims(self.ht, 2)
			else:
				self.H = functions.concat((self.H, functions.expand_dims(self.ht, 2)), axis=2)

		return self.H


	def reset_state(self):
		self.set_state(None, None, None, None)

	def set_state(self, ct, ht, H, contexts):
		self.ct = ct	# last cell state
		self.ht = ht	# last hidden state
		self.H = H		# all hidden states
		self.contexts = contexts
