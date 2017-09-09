import sys, os, json, pickle, math, chainer, uuid
import chainer.functions as F
import chainer.links as L
from six.moves import range
from chainer import Chain, serializers, initializers, variable, functions
sys.path.append(os.path.join("..", "..", ".."))
from asr.model.cnn import AcousticModel
from asr.stream import Stream
from asr.utils import to_dict, to_object
import asr.stream as nn

def build_model(config):
	vocab_size = 			config.vocab_size
	ndim_audio_features = 	config.ndim_audio_features
	ndim_h = 				config.ndim_h
	ndim_dense = 			config.ndim_dense
	kernel_size = 			config.kernel_size
	num_conv_layers = 		config.num_conv_layers
	dropout = 				config.dropout
	weightnorm = 			config.weightnorm
	wgain = 				config.wgain
	num_mel_filters = 		config.num_mel_filters
	architecture = 			config.architecture

	assert isinstance(vocab_size, int)
	assert isinstance(ndim_audio_features, int)
	assert isinstance(ndim_h, int)
	assert isinstance(ndim_dense, int)
	assert isinstance(kernel_size, (tuple, list))
	assert isinstance(num_conv_layers, int)
	assert isinstance(dropout, float)
	assert isinstance(weightnorm, bool)
	assert isinstance(wgain, (int, float))
	assert isinstance(num_mel_filters, int)
	assert isinstance(architecture, str)

	model = AcousticModel()
	pad = kernel_size[1] - 1
	kernel_height = int(math.ceil((num_mel_filters - 2) / 3))

	if architecture == "zhang":
		# first layer
		model.layer(
			nn.Convolution2D(ndim_audio_features, ndim_h * 2, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), weightnorm=weightnorm),
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
				nn.Convolution2D(in_channels, out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), weightnorm=weightnorm),
				lambda x: x[..., :-pad],
				nn.Maxout(2),
				nn.Dropout(dropout),
			)

		if num_conv_layers_wide > 0:
			in_out = [(ndim_h * 2, ndim_h * 4)] * num_conv_layers_narrow
			for in_channels, out_channels in in_out:
				model.layer(
					nn.Convolution2D(in_channels, out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), weightnorm=weightnorm),
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
			nn.Convolution2D(ndim_audio_features, ndim_h * 2, kernel_size, stride=1, pad=(0, kernel_size[1] - 1), weightnorm=weightnorm),
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
				nn.Convolution2D(in_channels, out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), weightnorm=weightnorm),
				lambda x: x[..., :-pad],
				nn.Maxout(2),
				nn.Dropout(dropout),
			)

		if num_conv_layers_wide > 0:
			in_out = [(ndim_h * 2, ndim_h * 4)] * num_conv_layers_narrow
			for in_channels, out_channels in in_out:
				model.layer(
					nn.Convolution2D(in_channels, out_channels, kernel_size, stride=1, pad=(1, kernel_size[1] - 1), weightnorm=weightnorm),
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
		for _ in range(num_conv_layers):
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
		for _ in range(num_conv_layers):
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
		for _ in range(num_conv_layers):
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
		for _ in range(num_conv_layers):
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
		