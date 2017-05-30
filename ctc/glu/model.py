# coding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, os, json, pickle, math
import chainer.functions as F
from six.moves import xrange
from chainer import Chain, serializers, initializers
sys.path.append("../../")
import glu as L

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
# Towards End-to-End Speech Recognition with Deep Convolutional Neural Networks
# https://arxiv.org/abs/1701.02720
class ZhangModel(Chain):
	def __init__(self, vocab_size, num_blocks, num_layers_per_block, num_fc_layers, ndim_features, ndim_h, kernel_size=(3, 5), dropout=0, weightnorm=False, wgain=1, ignore_label=None):
		super(ZhangModel, self).__init__()
		assert num_blocks > 0
		assert num_layers_per_block > 0
		self.vocab_size = vocab_size
		self.num_blocks = num_blocks
		self.num_layers_per_block = num_layers_per_block
		self.num_fc_layers = num_fc_layers
		self.ndim_h = ndim_h
		self.ndim_features = ndim_features
		self.kernel_size = kernel_size
		self.weightnorm = weightnorm
		self.dropout = dropout
		self.using_dropout = True if dropout > 0 else False
		self.wgain = wgain
		self.ignore_label = ignore_label

		num_layers = 0

		wstd = math.sqrt(wgain / ndim_features / kernel_size[0] / kernel_size[1])
		if weightnorm:
			input_conv = L.WeightnormConvolution2D(ndim_features, ndim_h, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), initialV=initializers.Normal(wstd))
		else:
			input_conv = L.Convolution2D(ndim_features, ndim_h, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), initialW=initializers.Normal(wstd))

		self.add_link("input_conv", input_conv)
		num_layers += 1

		for i in xrange(num_blocks * num_layers_per_block):
			self.add_link("glu{}".format(i), L.GLU(ndim_h, ndim_h, kernel_size=kernel_size, wgain=wgain, weightnorm=weightnorm))
			num_layers += 1

		for i in xrange(num_fc_layers - 1):
			self.add_link("fc{}".format(i), L.Linear(None, 512))
			num_layers += 1
		self.add_link("fc{}".format(num_fc_layers - 1), L.Linear(None, vocab_size))
		num_layers += 1

		# batch normalization
		self.running_mean = [None] * num_layers
		self.running_stddev = [None] * num_layers
		self.running_z = None

	def get_glu_layer(self, index):
		return getattr(self, "glu{}".format(index))

	def get_fc_layer(self, index):
		return getattr(self, "fc{}".format(index))

	def reset_state(self):
		for i in xrange(self.num_blocks * self.num_layers_per_block):
			self.get_glu_layer(i).reset_state()

	def _forward_glu_layer(self, layer_index, in_data):
		glu = self.get_glu_layer(layer_index)
		out_data = glu(in_data)
		return out_data

	def _forward_fc_layer(self, layer_index, in_data):
		fc = self.get_fc_layer(layer_index)
		out_data = fc(in_data)
		return out_data

	def normalize_layer(self, out_data, global_layer_index):
		xp = self.xp
		eps = 1e-4
		seq_length = out_data.shape[3]

		mean_batch = xp.mean(out_data.data, axis=0)[None, ...]
		stddev_batch = xp.std(out_data.data, axis=0)[None, ...] + eps

		running_mean, running_stddev, z = self.running_mean[global_layer_index], self.running_stddev[global_layer_index], self.running_z

		if z is None:
			z = xp.zeros((seq_length,), dtype=xp.float32).reshape((1, 1, 1, -1))
			self.running_z = z
		if running_mean is None:
			running_mean = xp.zeros((seq_length,), dtype=xp.float32).reshape((1, 1, 1, -1))
			self.running_mean[global_layer_index] = running_mean
		if running_stddev is None:
			running_stddev = xp.zeros((seq_length,), dtype=xp.float32).reshape((1, 1, 1, -1))
			self.running_stddev[global_layer_index] = running_stddev

		if seq_length > z.size:
			diff = seq_length - z.size
			z = xp.concatenate((z, xp.zeros((diff,), dtype=xp.float32).reshape(1, 1, 1, -1)), axis=3)
			self.running_z = z
		if seq_length > running_mean.size:
			diff = seq_length - running_mean.size
			running_mean = xp.concatenate((running_mean, xp.zeros((diff,), dtype=xp.float32).reshape(1, 1, 1, -1)), axis=3)
			self.running_mean[global_layer_index] = running_mean
		if seq_length > running_stddev.size:
			diff = seq_length - running_stddev.size
			running_stddev = xp.concatenate((running_stddev, xp.zeros((diff,), dtype=xp.float32).reshape(1, 1, 1, -1)), axis=3)
			self.running_stddev[global_layer_index] = running_stddev

		if seq_length < z.size:
			z = z[..., :seq_length]
			running_mean = running_mean[..., :seq_length]
			running_stddev = running_stddev[..., :seq_length]

		# print(running_mean.shape)
		# print(mean_batch.shape)
		# print(running_stddev.shape)
		# print(stddev_batch.shape)
		# print(z.shape)

		running_mean = running_mean * (z / (z + 1)) + mean_batch / (z + 1)			# 正規化定数が+1されることに注意
		running_stddev = running_stddev * (z / (z + 1)) + stddev_batch / (z + 1)	# 正規化定数が+1されることに注意

		out_data = (out_data - running_mean) / running_stddev

		self.running_mean[global_layer_index][..., :seq_length] = running_mean
		self.running_stddev[global_layer_index][..., :seq_length] = running_stddev

		return out_data

	def __call__(self, X, return_last=False):
		batchsize = X.shape[0]
		seq_length = X.shape[3]

		global_layer_index = 0

		### First Layer ###
		pad = self.kernel_size[1] - 1
		out_data = self.input_conv(X)[..., :-pad]
		out_data = F.relu(out_data)
		out_data = F.max_pooling_2d(out_data, ksize=(3, 1))
		out_data = self.normalize_layer(out_data, global_layer_index)
		global_layer_index += 1

		# residual connection
		residual_input = out_data

		### GLU Layers ###
		for layer_index in xrange(self.num_blocks * self.num_layers_per_block):
			out_data = self._forward_glu_layer(layer_index, out_data)
			out_data = self.normalize_layer(out_data, global_layer_index)
			if (layer_index + 1) % self.num_layers_per_block == 0:
				if self.using_dropout:
					out_data = F.dropout(out_data, ratio=self.dropout)
				out_data += residual_input
				residual_input = out_data
			global_layer_index += 1

		out_data = F.swapaxes(out_data, 1, 3)
		height = out_data.shape[2]
		out_data = F.reshape(out_data, (-1, self.ndim_h * height))

		if return_last:
			out_data = out_data[..., -1, None]
		
		# fully-connected layers
		for layer_index in xrange(self.num_fc_layers - 1):
			out_data = self._forward_fc_layer(layer_index, out_data)
			out_data = F.relu(out_data)
			if self.using_dropout:
				out_data = F.dropout(out_data, ratio=self.dropout)

		out_data = self._forward_fc_layer(self.num_fc_layers - 1, out_data)
		out_data = F.reshape(out_data, (batchsize, -1))
		out_data = F.split_axis(out_data, seq_length, axis=1)

		self.running_z += 1

		return out_data

	def _forward_layer_one_step(self, layer_index, in_data):
		glu = self.get_glu_layer(layer_index)
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

		out_data = self._forward_layer_one_step(0, enmbedding)[:, :, -ksize:]
		for layer_index in xrange(1, self.num_blocks * self.num_layers_per_block):
			out_data = self._forward_layer_one_step(layer_index, out_data)[:, :, -ksize:]
			if (layer_index + 1) % self.num_layers_per_block == 0:
				if self.using_dropout:
					out_data = F.dropout(out_data, ratio=self.dropout)
				out_data += residual_input
				residual_input = out_data

		out_data = out_data[:, :, -1, None]
		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (-1, self.ndim_h))
		Y = self.output_fc(out_data)

		return Y
