# coding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, os, json, pickle, math, chainer
import chainer.functions as F
import chainer.links as L
from six.moves import xrange
from chainer import Chain, serializers, initializers, variable, functions
sys.path.append("../../")
from convolution_2d import Convolution2D as WeightnormConvolution2D

def save_model(dirname, model):
	model_filename = dirname + "/model.hdf5"
	param_filename = dirname + "/params.json"

	try:
		os.mkdir(dirname)
	except:
		pass

	if os.path.isfile(model_filename):
		os.remove(model_filename)
	serializers.save_hdf5(model_filename, model)

	params = {
		"vocab_size": model.vocab_size,
		"ndim_embedding": model.ndim_embedding,
		"ndim_h": model.ndim_h,
		"num_layers_per_block": model.num_layers_per_block,
		"num_blocks": model.num_blocks,
		"kernel_size": model.kernel_size,
		"dropout": model.dropout,
		"weightnorm": model.weightnorm,
		"wgain": model.wgain,
		"ignore_label": model.ignore_label,
	}
	with open(param_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

def load_model(dirname):
	model_filename = dirname + "/model.hdf5"
	param_filename = dirname + "/params.json"

	if os.path.isfile(param_filename):
		print("loading {} ...".format(param_filename))
		with open(param_filename, "r") as f:
			try:
				params = json.load(f)
			except Exception as e:
				raise Exception("could not load {}".format(param_filename))

		qrnn = ZhangModel(**params)

		if os.path.isfile(model_filename):
			print("loading {} ...".format(model_filename))
			serializers.load_hdf5(model_filename, qrnn)

		return qrnn
	else:
		return None

def Convolution2D(in_channel, out_channel, ksize, stride=1, pad=0, initialW=None, weightnorm=False):
	if weightnorm:
		return WeightnormConvolution2D(in_channel, out_channel, ksize, stride=1, pad=pad, initialV=initialW)
	return L.Convolution2D(in_channel, out_channel, ksize, stride=1, pad=pad, initialW=initialW)

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

	def _normalize(self, x):
		size = x.shape[1] * x.shape[2]
		mean = functions.math.sum.sum(x, axis=(1, 2), keepdims=True) / size
		mean = functions.array.broadcast.broadcast_to(mean, x.shape)
		std = functions.math.sqrt.sqrt(functions.math.sum.sum(functions.math.square.square(x - mean), axis=(1, 2), keepdims=True) / size) + self.eps
		std = functions.array.broadcast.broadcast_to(std, x.shape)
		return (x - mean) / std

	def __call__(self, x):
		if self.gamma.data is None:
			self._initialize_params(x.size // x.shape[0])

		normalized = self._normalize(x)
		return functions.math.bias.bias(functions.math.scale.scale(normalized, self.gamma), self.beta)
		
# Towards End-to-End Speech Recognition with Deep Convolutional Neural Networks
# https://arxiv.org/abs/1701.02720
class ZhangModel(Chain):
	def __init__(self, vocab_size, num_conv_layers, num_fc_layers, ndim_audio_features, ndim_h, ndim_fc=1024, kernel_size=(3, 5), dropout=0, layernorm=False, weightnorm=False, residual=False, wgain=1, num_mel_filters=40):
		super(ZhangModel, self).__init__()
		assert num_conv_layers > 0
		assert num_fc_layers > 0
		assert ndim_audio_features > 0
		assert ndim_h > 0
		self.vocab_size = vocab_size
		self.num_conv_layers = num_conv_layers
		self.num_fc_layers = num_fc_layers
		self.ndim_audio_features = ndim_audio_features
		self.ndim_h = ndim_h
		self.ndim_fc = ndim_fc
		self.kernel_size = kernel_size
		self.weightnorm = weightnorm
		self.using_layernorm = True if layernorm else False
		self.dropout = dropout
		self.using_dropout = True if dropout > 0 else False
		self.using_residual = True if residual else False
		self.wgain = wgain

		wstd = math.sqrt(wgain / ndim_audio_features / kernel_size[0] / kernel_size[1])
		self.add_link("input_conv", Convolution2D(ndim_audio_features, ndim_h, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), initialW=initializers.Normal(wstd), weightnorm=weightnorm))
		self.add_link("input_conv_norm", LayerNormalization((ndim_h,)))

		wstd = math.sqrt(wgain / ndim_h / kernel_size[0] / kernel_size[1])
		for i in xrange(num_conv_layers):
			self.add_link("conv{}".format(i), Convolution2D(ndim_h, ndim_h, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.Normal(wstd), weightnorm=weightnorm))
			self.add_link("conv_norm{}".format(i), LayerNormalization((ndim_h,)))

		kernel_height = int(math.ceil((num_mel_filters - 2) / 3))
		if num_fc_layers == 1:
			self.add_link("fc0", Convolution2D(ndim_h, vocab_size, (kernel_height, 1), stride=1, pad=0))
			self.add_link("fc_norm0", LayerNormalization((vocab_size,)))
		else:
			self.add_link("fc0", Convolution2D(ndim_h, ndim_fc, (kernel_height, 1), stride=1, pad=0))
			self.add_link("fc_norm0", LayerNormalization((ndim_fc,)))
			for i in xrange(num_fc_layers - 2):
				self.add_link("fc{}".format(i + 1), Convolution2D(ndim_fc, ndim_fc, ksize=1, stride=1, pad=0))
				self.add_link("fc_norm{}".format(i + 1), LayerNormalization((ndim_fc,)))
			self.add_link("fc{}".format(num_fc_layers - 1), Convolution2D(ndim_fc, vocab_size, ksize=1, stride=1, pad=0))
			self.add_link("fc_norm{}".format(num_fc_layers - 1), LayerNormalization((vocab_size,)))

	def get_conv_layer(self, index):
		return getattr(self, "conv{}".format(index))

	def get_conv_norm_layer(self, index):
		return getattr(self, "conv_norm{}".format(index))

	def get_fc_layer(self, index):
		return getattr(self, "fc{}".format(index))

	def get_fc_norm_layer(self, index):
		return getattr(self, "fc_norm{}".format(index))

	def forward_conv_layer(self, layer_index, in_data):
		pad = self.kernel_size[1] - 1
		conv = self.get_conv_layer(layer_index)
		out_data = conv(in_data)[..., :-pad]
		return out_data

	def forward_fc_layer(self, layer_index, in_data):
		fc = self.get_fc_layer(layer_index)
		out_data = fc(in_data)
		return out_data

	# Layer Normalization
	# https://arxiv.org/abs/1607.06450
	def normalize_input_conv_layer(self, out_data):
		if self.using_layernorm == False:
			return out_data

		norm = getattr(self, "input_conv_norm")
		out_data = norm(out_data)
		return out_data

	# Layer Normalization
	# https://arxiv.org/abs/1607.06450
	def normalize_conv_layer(self, layer_index, out_data):
		if self.using_layernorm == False:
			return out_data

		norm = self.get_conv_norm_layer(layer_index)
		out_data = norm(out_data)
		return out_data

	# Layer Normalization
	# https://arxiv.org/abs/1607.06450
	def normalize_fc_layer(self, layer_index, out_data, batchsize, seq_length):
		if self.using_layernorm == False:
			return out_data

		norm = self.get_fc_norm_layer(layer_index)
		out_data = norm(out_data)
		return out_data

	def __call__(self, X, return_last=False, split_into_variables=True):
		batchsize = X.shape[0]
		seq_length = X.shape[3]

		### First Layer ###
		pad = self.kernel_size[1] - 1
		out_data = self.input_conv(X)[..., :-pad]
		out_data = self.normalize_input_conv_layer(out_data)
		out_data = F.relu(out_data)
		out_data = F.max_pooling_2d(out_data, ksize=(3, 1))

		# residual connection
		residual_input = out_data

		### Conv Layers ###
		for layer_index in xrange(self.num_conv_layers):
			out_data = self.forward_conv_layer(layer_index, out_data)
			out_data = F.relu(out_data)
			if self.using_dropout:
				out_data = F.dropout(out_data, ratio=self.dropout)
			if self.using_residual:
				out_data += residual_input
			out_data = self.normalize_conv_layer(layer_index, out_data)
			if self.using_residual:
				residual_input = out_data

		if return_last:
			out_data = out_data[..., -1, None]

		### Fully-connected Layers ###
		residual_input = 0
		for layer_index in xrange(self.num_fc_layers - 1):
			out_data = self.forward_fc_layer(layer_index, out_data)
			out_data = F.relu(out_data)
			if self.using_dropout:
				out_data = F.dropout(out_data, ratio=self.dropout)
			if self.using_residual:
				out_data += residual_input
			out_data = self.normalize_fc_layer(layer_index, out_data, batchsize, seq_length)
			if self.using_residual:
				residual_input = out_data

		# 最後のFC層には活性化関数を通さないので別に処理
		layer_index = self.num_fc_layers - 1
		out_data = self.forward_fc_layer(layer_index, out_data)
		# out_data = self.normalize_fc_layer(layer_index, out_data, batchsize, seq_length)

		xp = self.xp
		print(xp.mean(out_data.data), xp.std(out_data.data), xp.amax(out_data.data), xp.amin(out_data.data))


		# CTCでは同一時刻のRNN出力をまとめてVariableにする必要がある
		if split_into_variables:
			out_data = F.swapaxes(out_data, 1, 3)
			out_data = F.reshape(out_data, (batchsize, -1))
			out_data = F.split_axis(out_data, seq_length, axis=1)
		else:
			out_data = F.swapaxes(out_data, 1, 3)
			out_data = F.squeeze(out_data, axis=2)

		return out_data

	def forward_glu_layer_one_step(self, layer_index, in_data):
		glu = self.get_conv_layer(layer_index)
		out_data = glu.forward_one_step(in_data)
		return out_data

	def forward_one_step(self, X):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		ksize = self.kernel_size

		if seq_length < ksize:
			self.reset_state()
			return self.__call__(X, return_last=True)

		xt = X[:, -ksize:]
		enmbedding = self.embed(xt)
		enmbedding = F.swapaxes(enmbedding, 1, 2)
		residual_input = enmbedding if self.ndim_h == self.ndim_embedding else 0

		out_data = self.forward_glu_layer_one_step(0, enmbedding)[:, :, -ksize:]
		for layer_index in xrange(1, self.num_blocks * self.num_layers_per_block):
			out_data = self.forward_glu_layer_one_step(layer_index, out_data)[:, :, -ksize:]
			if (layer_index + 1) % self.num_layers_per_block == 0:
				if self.using_dropout:
					out_data = F.dropout(out_data, ratio=self.dropout)
				out_data += residual_input
				residual_input = out_data

		out_data = out_data[:, :, -1, None]
		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (-1, self.ndim_h))
		Y = self.output_fc(out_data)

		return Y
