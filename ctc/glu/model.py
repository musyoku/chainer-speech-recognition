import sys, os, json, pickle
import chainer.functions as F
from six.moves import xrange
from chainer import Chain, serializers
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

		qrnn = RNNModel(**params)

		if os.path.isfile(model_filename):
			print("loading {} ...".format(model_filename))
			serializers.load_hdf5(model_filename, qrnn)

		return qrnn
	else:
		return None

class RNNModel(Chain):
	def __init__(self, vocab_size, ndim_embedding, num_blocks, num_layers_per_block, ndim_h, kernel_size=4, dropout=0, weightnorm=False, wgain=1, ignore_label=None):
		super(RNNModel, self).__init__(
			dense=L.Linear(ndim_h, vocab_size),
		)
		assert num_blocks > 0
		assert num_layers_per_block > 0
		self.vocab_size = vocab_size
		self.ndim_embedding = ndim_embedding
		self.num_blocks = num_blocks
		self.num_layers_per_block = num_layers_per_block
		self.ndim_h = ndim_h
		self.kernel_size = kernel_size
		self.weightnorm = weightnorm
		self.dropout = dropout
		self.using_dropout = True if dropout > 0 else False
		self.wgain = wgain
		self.ignore_label = ignore_label

		self.add_link("glu0", L.GLU(ndim_embedding, ndim_h, kernel_size=kernel_size, wgain=wgain, weightnorm=weightnorm))
		for i in xrange(1, num_blocks * num_layers_per_block):
			self.add_link("glu{}".format(i), L.GLU(ndim_h, ndim_h, kernel_size=kernel_size, wgain=wgain, weightnorm=weightnorm))

	def get_glu_layer(self, index):
		return getattr(self, "glu{}".format(index))

	def reset_state(self):
		for i in xrange(self.num_blocks * self.num_layers_per_block):
			self.get_glu_layer(i).reset_state()

	def _forward_layer(self, layer_index, in_data):
		glu = self.get_glu_layer(layer_index)
		out_data = glu(in_data)
		return out_data

	def __call__(self, X, return_last=False):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		enmbedding = self.embed(X)
		enmbedding = F.swapaxes(enmbedding, 1, 2)
		residual_input = enmbedding if self.ndim_h == self.ndim_embedding else 0

		out_data = self._forward_layer(0, enmbedding)
		for layer_index in xrange(1, self.num_blocks * self.num_layers_per_block):
			out_data = self._forward_layer(layer_index, out_data)
			if (layer_index + 1) % self.num_layers_per_block == 0:
				if self.using_dropout:
					out_data = F.dropout(out_data, ratio=self.dropout)
				out_data += residual_input
				residual_input = out_data

		if return_last:
			out_data = out_data[:, :, -1, None]

		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (-1, self.ndim_h))
		Y = self.dense(out_data)

		return Y

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
		Y = self.dense(out_data)

		return Y
