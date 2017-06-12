from __future__ import division
from __future__ import print_function
from six.moves import xrange
import chainer, argparse, math, cupy
import numpy as np
from chainer import optimizers, cuda
from chainer import functions as F
from model import ZhangModel

BLANK = 0

def compute_character_error_rate(r, h):
	d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
	for i in xrange(len(r) + 1):
		for j in xrange(len(h) + 1):
			if i == 0: d[0][j] = j
			elif j == 0: d[i][0] = i
	for i in xrange(1, len(r) + 1):
		for j in xrange(1, len(h) + 1):
			if r[i-1] == h[j-1]:
				d[i][j] = d[i-1][j-1]
			else:
				substitute = d[i-1][j-1] + 1
				insert = d[i][j-1] + 1
				delete = d[i-1][j] + 1
				d[i][j] = min(substitute, insert, delete)
	return float(d[len(r)][len(h)]) / len(r)

def generate_data():
	x_batch = np.random.normal(size=(args.dataset_size, 3, 40, args.sequence_length))
	t_batch = np.zeros((args.dataset_size, args.sequence_length), dtype=np.int32)
	true_data = np.random.normal(size=(args.vocab_size, 3, 40))
	for data_idx in xrange(len(x_batch)):
		indices = np.random.choice(np.arange(args.sequence_length), size=args.true_sequence_length, replace=False)
		tokens = np.random.choice(np.arange(1, args.vocab_size), size=args.true_sequence_length)
		for t, token in zip(indices, tokens): 
			x_batch[data_idx, ..., t] = true_data[token]
			t_batch[data_idx, t] = token
	t_batch = t_batch[t_batch > 0].reshape((args.dataset_size, args.true_sequence_length))
	return x_batch, t_batch

def main():
	model = ZhangModel(args.vocab_size, args.num_conv_layers, args.num_fc_layers, 3, args.ndim_h)
	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	train_data, train_labels = generate_data()
	total_loop = int(math.ceil(len(train_data) / args.batchsize))
	train_indices = np.arange(len(train_data), dtype=int)

	xp = model.xp
	x_length_batch = xp.full((args.batchsize,), args.sequence_length, dtype=xp.int32)
	t_length_batch = xp.full((args.batchsize,), args.true_sequence_length, dtype=xp.int32)

	# optimizer
	optimizer = optimizers.Adam(args.learning_rate, 0.9)
	optimizer.setup(model)
	

	for epoch in xrange(1, args.total_epoch + 1):
		# train loop
		sum_loss = 0
		with chainer.using_config("train", True):
			for itr in xrange(1, total_loop + 1):
				# sample minibatch
				np.random.shuffle(train_indices)
				x_batch = train_data[train_indices[:args.batchsize]]
				t_batch = train_labels[train_indices[:args.batchsize]]

				# GPU
				if xp is cupy:
					x_batch = cuda.to_gpu(x_batch.astype(xp.float32))
					t_batch = cuda.to_gpu(t_batch.astype(xp.int32))

				# forward
				y_batch = model(x_batch)	# list of variables

				# compute loss
				loss = F.connectionist_temporal_classification(y_batch, t_batch, BLANK, x_length_batch, t_length_batch)
				optimizer.update(lossfun=lambda: loss)

				sum_loss += float(loss.data)

		# evaluate
		with chainer.using_config("train", False):
			# sample minibatch
			np.random.shuffle(train_indices)
			x_batch = train_data[train_indices[:args.batchsize]]
			t_batch = train_labels[train_indices[:args.batchsize]]

			# GPU
			if xp is cupy:
				x_batch = cuda.to_gpu(x_batch.astype(xp.float32))
				t_batch = cuda.to_gpu(t_batch.astype(xp.int32))

			# forward
			y_batch = model(x_batch, split_into_variables=False)
			y_batch = xp.argmax(y_batch.data, axis=2)

			average_error = 0
			for input_sequence, argmax_sequence, true_sequence in zip(x_batch, y_batch, t_batch):
				pred_seqence = []
				for token in argmax_sequence:
					if token == BLANK:
						continue
					pred_seqence.append(int(token))
				print("true:", true_sequence, "pred:", pred_seqence)
				error = compute_character_error_rate(true_sequence.tolist(), pred_seqence)
				average_error += error
			print("CER: {} - loss: {} - lr: {:.4e}".format(int(average_error / args.batchsize * 100), sum_loss / total_loop, optimizer.alpha))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--total-epoch", "-epoch", type=int, default=100)
	parser.add_argument("--batchsize", "-b", type=int, default=32)
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.01)
	parser.add_argument("--vocab-size", "-vocab", type=int, default=50)
	parser.add_argument("--num-conv-layers", "-conv", type=int, default=1)
	parser.add_argument("--num-fc-layers", "-fc", type=int, default=1)
	parser.add_argument("--ndim-h", "-nh", type=int, default=128)
	parser.add_argument("--ndim-fc", "-nf", type=int, default=128)
	parser.add_argument("--true-sequence-length", "-tseq", type=int, default=5)
	parser.add_argument("--sequence-length", "-seq", type=int, default=30)
	parser.add_argument("--dataset-size", "-size", type=int, default=500)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	args = parser.parse_args()
	main()