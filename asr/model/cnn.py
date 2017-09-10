import sys, os, json, pickle, math, chainer, uuid
import chainer.functions as F
import chainer.links as L
from chainer import Chain, serializers, initializers, variable, functions
from ..stream import Stream
from ..utils import to_dict, to_object, dump_dict, printb, _set
from .. import stream as nn
from . import base

class Configuration(base.Configuration):
	def __init__(self):
		super().__init__()
		self.vocab_size = -1
		self.ndim_audio_features = 40
		self.ndim_h = 128
		self.ndim_dense = 256
		self.num_conv_layers = 5
		self.kernel_size = (3, 5)
		self.dropout = 0
		self.weightnorm = False
		self.wgain = 1
		self.architecture = "zhang"

	def save(filename, overwrite=False):
		assert self.vocab_size > 0

		if os.path.isfile(filename) and overwrite is False:
			return

		params = to_dict(self)
		
		with open(filename, "w") as f:
			json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

	def load(filename):
		if os.path.isfile(filename):
			print("Loading {} ...".format(filename))
			with open(filename, "r") as f:
				try:
					params = json.load(f)
				except Exception as e:
					raise Exception("could not load {}".format(filename))

			config = Configuration()
			_set(config, params)
			return config
		else:
			return None


def configure():
	return Configuration()
		
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

	def save(self, filename):
		tmp_filename = str(uuid.uuid4())
		serializers.save_hdf5(tmp_filename, model)
		if os.path.isfile(filename):
			os.remove(filename)
		os.rename(tmp_filename, filename)

	def load(filename):
		if os.path.isfile(filename):
			print("Loading {} ...".format(filename))
			serializers.load_hdf5(filename, self)
			return True
		return False
		