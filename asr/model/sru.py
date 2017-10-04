import sys, os, json, pickle, math, chainer, uuid
import chainer.functions as F
import chainer.links as L
from chainer import Chain, serializers, initializers, variable, functions
from ..utils import to_dict, to_object, dump_dict, printb, _set
from .. import nn
from . import base

class Configuration(base.Configuration):
	def __init__(self):
		super().__init__()
		self.vocab_size = -1
		self.ndim_audio_features = 40
		self.ndim_conv = 64
		self.ndim_rnn = 128
		self.ndim_dense = 256
		self.num_rnn_layers = 2
		self.kernel_size = (3, 5)
		self.dropout = 0

	def save(self, filename):
		assert self.vocab_size > 0
		super().save(filename)

def configure():
	return Configuration()
		
class AcousticModel(nn.Module):
	def save(self, filename):
		tmp_filename = str(uuid.uuid4())
		serializers.save_hdf5(tmp_filename, self)
		if os.path.isfile(filename):
			os.remove(filename)
		os.rename(tmp_filename, filename)

	def load(self, filename):
		if os.path.isfile(filename):
			print("Loading {} ...".format(filename))
			serializers.load_hdf5(filename, self)
			return True
		return False
		