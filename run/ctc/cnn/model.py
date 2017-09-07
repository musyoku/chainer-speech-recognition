# coding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, os, json, pickle, math, chainer, uuid
import chainer.functions as F
import chainer.links as L
from six.moves import xrange
from chainer import Chain, serializers, initializers, variable, functions
sys.path.append(os.path.join("..", "..", ".."))
from asr.stream import Stream
import asr.stream as nn

def save_model(filename, model):
	tmp_filename = str(uuid.uuid4())
	serializers.save_hdf5(tmp_filename, model)
	if os.path.isfile(filename):
		os.remove(filename)
	os.rename(tmp_filename, filename)

def save_config(filename, config, overwrite=False):
	if os.path.isfile(filename) and overwrite is False:
		return

	params = {
		"vocab_size": config.vocab_size,
		"ndim_dense": config.ndim_dense,
		"ndim_h": config.ndim_h,
		"ndim_audio_features": config.ndim_audio_features,
		"kernel_size": config.kernel_size,
		"dropout": config.dropout,
		"weightnorm": config.weightnorm,
		"architecture": config.architecture,
		"num_conv_layers": config.num_conv_layers,
		"wgain": config.wgain,
	}
	with open(filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

def load_config(filename):
	if os.path.isfile(filename):
		print("loading {} ...".format(filename))
		with open(filename, "r") as f:
			try:
				params = json.load(f)
			except Exception as e:
				raise Exception("could not load {}".format(filename))

		return params
	else:
		return None

def load_model(model_filename, config_filename):
	if os.path.isfile(model_filename):
		config = load_config(config_filename)
		assert config is not None, "{} not found.".format(config_filename)
		model = build_model(**config)

		if os.path.isfile(model_filename):
			print("loading {} ...".format(model_filename))
			serializers.load_hdf5(model_filename, model)
			
		return model
	else:
		return None

def build_model(vocab_size, ndim_audio_features=3, ndim_h=128, ndim_dense=320, kernel_size=(3, 5), num_conv_layers=4, dropout=0, weightnorm=False, wgain=1, num_mel_filters=40, architecture="zhang"):
	model = AcousticModel()
	pad = kernel_size[1] - 1
	kernel_height = int(math.ceil((num_mel_filters - 2) / 3))

	if architecture == "zhang":
		# first layer
		model.layer(
			nn.Convolution2D(ndim_audio_features, ndim_h * 2, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_audio_features / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
			lambda x: x[..., :-pad],
			nn.Maxout(2),
			nn.Dropout(dropout),
			nn.MaxPooling2D(ksize=(3, 1)),
		)
		# conv layers
		num_conv_layers_narrow = min(num_conv_layers, 4)
		num_conv_layers_wide = max(0, num_conv_layers - 4)
		in_out = [(ndim_h, ndim_h * 2)] * num_conv_layers_narrow
		if num_conv_layers_wide > 0:
			in_out[-1] = (ndim_h, ndim_h * 4)

		for in_channels, out_channels in in_out:
			model.layer(
				nn.Convolution2D(in_channels, out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_h / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
				lambda x: x[..., :-pad],
				nn.Maxout(2),
				nn.Dropout(dropout),
			)

		if num_conv_layers_wide > 0:
			in_out = [(ndim_h * 2, ndim_h * 4)] * num_conv_layers_narrow
			for in_channels, out_channels in in_out:
				model.layer(
					nn.Convolution2D(in_channels, out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_h / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
					lambda x: x[..., :-pad],
					nn.Maxout(2),
					nn.Dropout(dropout),
				)

		# dense layers
		model.layer(
			nn.Convolution2D(in_out[-1][0], ndim_dense * 2, ksize=(kernel_height, 1), stride=1, pad=0, weightnorm=weightnorm),
			nn.Maxout(2),
			nn.Dropout(dropout),
		)
		model.layer(
			nn.Convolution2D(ndim_dense, ndim_dense * 2, ksize=1, stride=1, pad=0, weightnorm=weightnorm),
			nn.Maxout(2),
			nn.Dropout(dropout),
		)
		model.layer(
			nn.Convolution2D(ndim_dense, vocab_size, ksize=1, stride=1, pad=0, weightnorm=weightnorm),
			nn.LayerNormalization(None),
		)
		return model

	if architecture == "zhang+fc_relu":
		# first layer
		model.layer(
			nn.Convolution2D(ndim_audio_features, ndim_h * 2, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_audio_features / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
			lambda x: x[..., :-pad],
			nn.Maxout(2),
			nn.Dropout(dropout),
			nn.MaxPooling2D(ksize=(3, 1)),
		)
		# conv layers
		num_conv_layers_narrow = min(num_conv_layers, 4)
		num_conv_layers_wide = max(0, num_conv_layers - 4)
		in_out = [(ndim_h, ndim_h * 2)] * num_conv_layers_narrow
		if num_conv_layers_wide > 0:
			in_out[-1] = (ndim_h, ndim_h * 4)

		for in_channels, out_channels in in_out:
			model.layer(
				nn.Convolution2D(in_channels, out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_h / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
				lambda x: x[..., :-pad],
				nn.Maxout(2),
				nn.Dropout(dropout),
			)

		if num_conv_layers_wide > 0:
			in_out = [(ndim_h * 2, ndim_h * 4)] * num_conv_layers_narrow
			for in_channels, out_channels in in_out:
				model.layer(
					nn.Convolution2D(in_channels, out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_h / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
					lambda x: x[..., :-pad],
					nn.Maxout(2),
					nn.Dropout(dropout),
				)

		# dense layers
		model.layer(
			nn.Convolution2D(in_out[-1][0], ndim_dense, ksize=(kernel_height, 1), stride=1, pad=0, weightnorm=weightnorm),
			nn.ReLU(),
			nn.Dropout(dropout),
		)
		model.layer(
			nn.Convolution2D(ndim_dense, ndim_dense, ksize=1, stride=1, pad=0, weightnorm=weightnorm),
			nn.ReLU(),
			nn.Dropout(dropout),
		)
		model.layer(
			nn.Convolution2D(ndim_dense, vocab_size, ksize=1, stride=1, pad=0, weightnorm=weightnorm),
			nn.LayerNormalization(None),
		)
		return model

	if architecture == "zhang+residual":
		# first layer
		model.layer(
			nn.Convolution2D(ndim_audio_features, ndim_h * 2, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_audio_features / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
			lambda x: x[..., :-pad],
			nn.Maxout(2),
			nn.Dropout(dropout),
			nn.MaxPooling2D(ksize=(3, 1)),
		)

		# conv layers
		num_conv_layers_narrow = min(num_conv_layers, 4)
		num_conv_layers_wide = max(0, num_conv_layers - 4)
		in_out = [(ndim_h, ndim_h * 2)] * num_conv_layers_narrow
		if num_conv_layers_wide > 0:
			in_out[-1] = (ndim_h, ndim_h * 4)

		for layer_idx, (in_channels, out_channels) in enumerate(in_out):
			if layer_idx == len(in_out) - 1:
				model.layer(
					nn.Convolution2D(in_channels, out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_h / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
					lambda x: x[..., :-pad],
					nn.Maxout(2),
					nn.Dropout(dropout),
				)
			else:
				model.layer(
					nn.Residual(
						nn.Convolution2D(in_channels, out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_h / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
						lambda x: x[..., :-pad],
						nn.Maxout(2),
						nn.Dropout(dropout),
					)
				)

		if num_conv_layers_wide > 0:
			in_out = [(ndim_h * 2, ndim_h * 4)] * num_conv_layers_narrow
			for in_channels, out_channels in in_out:
				model.layer(
					nn.Residual(
						nn.Convolution2D(in_channels, out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_h / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
						lambda x: x[..., :-pad],
						nn.Maxout(2),
						nn.Dropout(dropout),
					)
				)
				
		# dense layers
		model.layer(
			nn.Convolution2D(in_out[-1][0], ndim_dense * 2, ksize=(kernel_height, 1), stride=1, pad=0, weightnorm=weightnorm),
			nn.Maxout(2),
			nn.Dropout(dropout),
		)
		model.layer(
			nn.Convolution2D(ndim_dense, ndim_dense * 2, ksize=1, stride=1, pad=0, weightnorm=weightnorm),
			nn.Maxout(2),
			nn.Dropout(dropout),
		)
		model.layer(
			nn.Convolution2D(ndim_dense, vocab_size, ksize=1, stride=1, pad=0, weightnorm=weightnorm),
			nn.LayerNormalization(None),
		)
		return model

	if architecture == "zhang+layernorm":
		# first layer
		model.layer(
			nn.Convolution2D(ndim_audio_features, ndim_h * 2, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_audio_features / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
			lambda x: x[..., :-pad],
			nn.LayerNormalization(None),
			nn.Maxout(2),
			nn.Dropout(dropout),
			nn.MaxPooling2D(ksize=(3, 1)),
		)
		# conv layers
		for _ in xrange(num_conv_layers):
			model.layer(
				nn.Convolution2D(ndim_h, ndim_h * 2, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_h / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
				lambda x: x[..., :-pad],
				nn.LayerNormalization(None),
				nn.Maxout(2),
				nn.Dropout(dropout),
			)
		# dense layers
		model.layer(
			nn.Convolution2D(ndim_h, ndim_dense * 2, ksize=(kernel_height, 1), stride=1, pad=0, weightnorm=weightnorm),
			nn.LayerNormalization(None),
			nn.Maxout(2),
			nn.Dropout(dropout),
		)
		model.layer(
			nn.Convolution2D(ndim_dense, vocab_size, ksize=1, stride=1, pad=0, weightnorm=weightnorm),
			nn.LayerNormalization(None),
		)
		return model

	if architecture == "glu":
		# first layer
		model.layer(
			nn.Convolution2D(ndim_audio_features, ndim_h * 2, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_audio_features / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
			lambda x: x[..., :-pad],
			# nn.LayerNormalization(None),
			nn.Maxout(2),
			nn.Dropout(dropout),
			nn.MaxPooling2D(ksize=(3, 1)),
		)
		# conv layers
		for _ in xrange(num_conv_layers):
			model.layer(
				nn.GLU(ndim_h, ndim_h, kernel_size, pad=(1, kernel_size[1] - 1), weightnorm=weightnorm),
				# nn.LayerNormalization(None),
				nn.Dropout(dropout),
			)
		# dense layers
		model.layer(
			nn.GLU(ndim_h, ndim_dense, ksize=(kernel_height, 1), pad=0, weightnorm=weightnorm),
			# nn.LayerNormalization(None),
			nn.Dropout(dropout),
		)
		model.layer(
			nn.Convolution2D(ndim_dense, vocab_size, ksize=1, stride=1, pad=0, weightnorm=weightnorm),
			nn.LayerNormalization(None),
		)
		return model

	if architecture == "relu+layernorm":
		# first layer
		model.layer(
			nn.Convolution2D(ndim_audio_features, ndim_h, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_audio_features / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
			lambda x: x[..., :-pad],
			nn.LayerNormalization(None),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.MaxPooling2D(ksize=(3, 1)),
		)
		# conv layers
		for _ in xrange(num_conv_layers):
			model.layer(
				nn.Convolution2D(ndim_h, ndim_h, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_h / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
				lambda x: x[..., :-pad],
				nn.LayerNormalization(None),
				nn.ReLU(),
				nn.Dropout(dropout),
			)
		# dense layers
		model.layer(
			nn.Convolution2D(ndim_h, ndim_dense, ksize=(kernel_height, 1), stride=1, pad=0, weightnorm=weightnorm),
			nn.LayerNormalization(None),
			nn.ReLU(),
			nn.Dropout(dropout),
		)
		model.layer(
			nn.Convolution2D(ndim_dense, vocab_size, ksize=1, stride=1, pad=0, weightnorm=weightnorm),
			nn.LayerNormalization(None),
		)
		return model

	if architecture == "relu+layernorm+residual":
		# first layer
		model.layer(
			nn.Convolution2D(ndim_audio_features, ndim_h, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_audio_features / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
			lambda x: x[..., :-pad],
			nn.LayerNormalization(None),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.MaxPooling2D(ksize=(3, 1)),
		)
		# conv layers
		for _ in xrange(num_conv_layers):
			model.layer(
				nn.Residual(
					nn.LayerNormalization(None),
					nn.ReLU(),
					nn.Dropout(dropout),
					nn.Convolution2D(ndim_h, ndim_h, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), initialW=initializers.Normal(math.sqrt(wgain / ndim_h / kernel_size[0] / kernel_size[1])), weightnorm=weightnorm),
					lambda x: x[..., :-pad],
				)
			)
		# dense layers
		model.layer(
			nn.Convolution2D(ndim_h, ndim_dense, ksize=(kernel_height, 1), stride=1, pad=0, weightnorm=weightnorm),
			nn.LayerNormalization(None),
			nn.ReLU(),
			nn.Dropout(dropout),
		)
		model.layer(
			nn.Convolution2D(ndim_dense, vocab_size, ksize=1, stride=1, pad=0, weightnorm=weightnorm),
			nn.LayerNormalization(None),
		)
		return model
	raise NotImplementedError()
		
# Towards End-to-End Speech Recognition with Deep Convolutional Neural Networks
# https://arxiv.org/abs/1701.02720
class AcousticModel(Stream):
	def __call__(self, x, split_into_variables=True):
		batchsize = x.shape[0]
		seq_length = x.shape[3]

		out_data = super(AcousticModel, self).__call__(x)
		assert out_data.shape[3] == seq_length

		# CTCでは同一時刻のRNN出力をまとめてVariableにする必要がある
		if split_into_variables:
			out_data = F.swapaxes(out_data, 1, 3)
			out_data = F.reshape(out_data, (batchsize, -1))
			out_data = F.split_axis(out_data, seq_length, axis=1)
		else:
			out_data = F.swapaxes(out_data, 1, 3)
			out_data = F.squeeze(out_data, axis=2)

		return out_data