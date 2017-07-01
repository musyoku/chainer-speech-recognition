from __future__ import division
from __future__ import print_function
from six.moves import xrange
import chainer, argparse, math, cupy, sys, os, time
import numpy as np
from chainer import optimizers, cuda, serializers
from chainer import functions as F
sys.path.append("../../")
from train import get_optimizer, get_current_learning_rate
from model import AcousticModel, load_model, save_model, build_model
from ctc import connectionist_temporal_classification

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
	t_length_batch = np.zeros((args.dataset_size,), dtype=np.int32)
	x_length_batch = np.zeros((args.dataset_size,), dtype=np.int32)

	for data_idx in xrange(len(x_batch)):
		num_tokens = np.random.randint(1, high=args.true_sequence_length + 1, size=1)
		x_length = np.random.randint(num_tokens * 2 + 1, high=args.sequence_length + 1, size=1)
		tokens = np.random.choice(np.arange(1, args.vocab_size), size=num_tokens)

		indices = np.random.choice(np.arange(x_length), size=num_tokens, replace=False)
		for token_idx, token in zip(indices, tokens): 
			x_batch[data_idx, ..., token_idx] = true_data[token]
			t_batch[data_idx, token_idx] = token

		t = t_batch[data_idx]
		t = t[t > 0]
		t_batch[data_idx] = BLANK
		t_batch[data_idx, :len(t)] = t
		
		valid_tokens = t_batch[data_idx]
		valid_tokens = valid_tokens[valid_tokens != BLANK]
		num_trans_same_label = np.count_nonzero(valid_tokens == np.roll(valid_tokens, 1))
		x_length += max(0, num_tokens + num_trans_same_label - x_length)
		if x_length > args.sequence_length:
			print(num_tokens)
			print(num_trans_same_label)
			print(t_batch[data_idx])
			print(x_length)
			print(args.sequence_length)
			t_batch[data_idx, -(x_length - args.sequence_length):] = BLANK
			print(t_batch[data_idx])
			raise Exception()

		t_length_batch[data_idx] = num_tokens
		x_length_batch[data_idx] = x_length

	return x_batch, t_batch[..., :args.true_sequence_length], x_length_batch, t_length_batch

def main():
	np.random.seed(33)
	# np.set_printoptions(precision=3, suppress=True)

	model_filename = "model.hdf5"
	model = build_model(vocab_size=args.vocab_size, ndim_h=args.ndim_h, ndim_dense=320,
		 kernel_size=(3, 5), dropout=args.dropout, weightnorm=args.weightnorm, architecture=args.architecture)
	if os.path.isfile(model_filename):
		print("loading {} ...".format(model_filename))
		serializers.load_hdf5(model_filename, model)
	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	train_data, train_labels, train_data_length, train_label_length = generate_data()
	total_loop = int(math.ceil(len(train_data) / args.batchsize))
	train_indices = np.arange(len(train_data), dtype=int)

	xp = model.xp

	# optimizer
	optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(1))
	

	for epoch in xrange(1, args.total_epoch + 1):
		# train loop
		sum_loss = 0
		start_time = time.time()
		with chainer.using_config("debug", True):
			for itr in xrange(1, total_loop + 1):
				# sample minibatch
				np.random.shuffle(train_indices)
				x_batch = train_data[train_indices[:args.batchsize]]
				t_batch = train_labels[train_indices[:args.batchsize]]
				x_length_batch = train_data_length[train_indices[:args.batchsize]]
				t_length_batch = train_label_length[train_indices[:args.batchsize]]

				x_max_length = np.amax(x_length_batch)
				x_batch = x_batch[..., :x_max_length]

				# GPU
				if xp is cupy:
					x_batch = cuda.to_gpu(x_batch.astype(xp.float32))
					t_batch = cuda.to_gpu(t_batch.astype(xp.int32))
					x_length_batch = cuda.to_gpu(x_length_batch.astype(xp.int32))
					t_length_batch = cuda.to_gpu(t_length_batch.astype(xp.int32))


				# t_batch[0, 1] = 2
				# if t_length_batch.size == 4:
				# 	t_batch[1, 1:] = 0
				# 	t_length_batch[1] = 1
				# 	t_batch[2, 2] = 0
				# 	t_length_batch[2] = 2
				# 	t_batch[3, 1] = 2
				# print(x_length_batch)
				# print(t_batch)
				# print(t_length_batch)

				# forward
				y_batch = model(x_batch)	# list of variables

				# compute loss
				loss = connectionist_temporal_classification(y_batch, t_batch, BLANK, x_length_batch, t_length_batch, reduce="no", softmax_scale=math.sqrt(1))
				loss_value = float(xp.sum(loss.data))
				if loss_value != loss_value:
					if os.path.isfile(model_filename):
						os.remove(model_filename)
					serializers.save_hdf5(model_filename, model)

				assert loss_value == loss_value
				optimizer.update(lossfun=lambda: F.sum(loss))

				sum_loss += loss_value

				sys.stdout.write("\riteration {}/{}".format(itr, total_loop))
				sys.stdout.flush()

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
				target_sequence = []
				for token in true_sequence:
					if token == BLANK:
						continue
					target_sequence.append(int(token))
				pred_seqence = []
				for token in argmax_sequence:
					if token == BLANK:
						continue
					pred_seqence.append(int(token))
				print("true:", target_sequence, "pred:", pred_seqence)
				error = compute_character_error_rate(target_sequence, pred_seqence)
				average_error += error
			print("CER: {} - loss: {} - lr: {:.4e}".format(int(average_error / args.batchsize * 100), sum_loss / total_loop, get_current_learning_rate(optimizer)))

		print("elapsed:", time.time() - start_time)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--total-epoch", "-epoch", type=int, default=100)
	parser.add_argument("--batchsize", "-b", type=int, default=32)
	parser.add_argument("--vocab-size", "-vocab", type=int, default=83)
	parser.add_argument("--num-conv-layers", "-conv", type=int, default=1)
	parser.add_argument("--num-fc-layers", "-fc", type=int, default=1)
	parser.add_argument("--ndim-h", "-nh", type=int, default=128)
	parser.add_argument("--ndim-fc", "-nf", type=int, default=128)
	parser.add_argument("--true-sequence-length", "-tseq", type=int, default=10)
	parser.add_argument("--sequence-length", "-seq", type=int, default=100)
	parser.add_argument("--dataset-size", "-size", type=int, default=500)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--dropout", "-dropout", type=float, default=0)
	parser.add_argument("--weightnorm", "-weightnorm", default=False, action="store_true")
	parser.add_argument("--layernorm", "-layernorm", default=False, action="store_true")
	parser.add_argument("--residual", "-residual", default=False, action="store_true")
	parser.add_argument("--architecture", "-arch", type=str, default="zhang+layernorm")
	parser.add_argument("--optimizer", "-opt", type=str, default="adam")
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.001)
	parser.add_argument("--momentum", "-mo", type=float, default=0.9)
	args = parser.parse_args()
	main()