import sys, os, json, pickle, math, chainer, uuid
import chainer.functions as F
import chainer.links as L
from six.moves import range
from chainer import Chain, serializers, initializers, variable, functions
from asr.model.sru import AcousticModel
from asr.utils import to_dict, to_object
import asr.nn as nn

class Model(AcousticModel):
	def __init__(self, config):
		super(Model, self).__init__()
		vocab_size = 			config.vocab_size
		ndim_audio_features = 	config.ndim_audio_features
		ndim_conv = 			config.ndim_conv
		ndim_rnn = 				config.ndim_rnn
		ndim_dense = 			config.ndim_dense
		kernel_size = 			config.kernel_size
		num_rnn_layers = 		config.num_rnn_layers
		dropout = 				config.dropout
		num_mel_filters = 		config.num_mel_filters

		assert isinstance(vocab_size, int)
		assert isinstance(ndim_audio_features, int)
		assert isinstance(ndim_rnn, int)
		assert isinstance(kernel_size, (tuple, list))
		assert isinstance(ndim_dense, int)
		assert isinstance(num_rnn_layers, int)
		assert isinstance(dropout, (float, int))
		assert isinstance(num_mel_filters, int)

		self.num_rnn_layers = num_rnn_layers
		self.reset_state()

		with self.init_scope():
			# first layer
			pad = kernel_size[1] - 1
			conv_blocks = nn.Module()
			conv_blocks.add(
				nn.Convolution2D(ndim_audio_features, ndim_conv * 2, kernel_size, stride=1, pad=(0, kernel_size[1] - 1)),
				lambda x: x[..., :-pad],
				nn.Maxout(2),
				nn.LayerNormalization(),
				nn.MaxPooling2D(ksize=(3, 1)),
				nn.Dropout(dropout),
			)
			conv_blocks.add(
				nn.Convolution2D(ndim_conv, ndim_conv * 2, kernel_size, stride=1, pad=(0, kernel_size[1] - 1)),
				lambda x: x[..., :-pad],
				nn.Maxout(2),
				nn.LayerNormalization(),
				nn.MaxPooling2D(ksize=(2, 1)),
				nn.Dropout(dropout),
			)
			conv_blocks.add(
				nn.Convolution2D(ndim_conv, ndim_rnn * 2, kernel_size, stride=1, pad=(0, kernel_size[1] - 1)),
				lambda x: x[..., :-pad],
				nn.Maxout(2),
				nn.LayerNormalization(),
				nn.MaxPooling2D(ksize=(2, 1)),
				nn.Dropout(dropout),
			)
			self.conv_blocks = conv_blocks

			# rnn layers
			rnn_blocks = nn.Module()
			for i in range(num_rnn_layers):
				rnn_blocks.add(
					nn.SRU(None),
					nn.Dropout(dropout),
				)
			self.rnn_blocks = rnn_blocks

			# dense layers
			dense_blocks = nn.Module()
			dense_blocks.add(
				nn.Convolution1D(None, ndim_dense * 2),
				nn.Maxout(2),
				nn.LayerNormalization(),
				nn.Dropout(dropout),
			)
			dense_blocks.add(
				nn.Convolution1D(ndim_dense, ndim_dense * 2),
				nn.Maxout(2),
				nn.LayerNormalization(),
				nn.Dropout(dropout),
			)
			dense_blocks.add(
				nn.Convolution1D(ndim_dense, vocab_size),
				nn.LayerNormalization(),
			)
			self.dense_blocks = dense_blocks

	def reset_state(self):
		self.contexts = [None] * self.num_rnn_layers

	def __call__(self, x, split_into_variables=True, discard_context=False):
		batchsize = x.shape[0]
		seq_length = x.shape[3]

		# conv
		out_data = self.conv_blocks(x)
		out_data = functions.reshape(out_data, (batchsize, -1, seq_length))

		# rnn
		for index, blocks in enumerate(self.rnn_blocks.blocks):
			sru = blocks[0]
			dropout = blocks[1] if len(blocks) == 2 else None
			hidden, cell, context = sru(out_data, self.contexts[index])
			if discard_context is False:
				self.contexts[index] = context
			if dropout is not None:
				out_data = dropout(out_data)

		# import numpy as np
		# np.set_printoptions(suppress=True)
		# print(out_data)

		# fc
		out_data = self.dense_blocks(out_data)
		assert out_data.shape[2] == seq_length

		# CTCでは同一時刻のRNN出力をまとめてVariableにする必要がある
		if split_into_variables:
			out_data = F.swapaxes(out_data, 1, 2)
			out_data = F.reshape(out_data, (batchsize, -1))
			out_data = F.split_axis(out_data, seq_length, axis=1)
		else:
			out_data = F.swapaxes(out_data, 1, 2)

		return out_data