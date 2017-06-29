# coding: utf8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, argparse, time, cupy, math, os
import chainer
import numpy as np
import chainer.functions as F
from chainer import optimizers, cuda, serializers
sys.path.append("../../")
from dataset import get_minibatch, get_vocab, load_buckets, get_duration_seconds
from model import load_model, save_model, build_model, ZhangModel
from ctc import connectionist_temporal_classification

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"
	MOVE = "\033[1A"
	LEFT = "\033[G"

def print_bold(str):
	print(stdout.BOLD + str + stdout.END)

def get_current_learning_rate(opt):
	if isinstance(opt, optimizers.NesterovAG):
		return opt.lr
	if isinstance(opt, optimizers.Adam):
		return opt.alpha
	raise NotImplementationError()

def get_optimizer(name, lr, momentum):
	if name == "nesterov":
		return optimizers.NesterovAG(lr=lr, momentum=momentum)
	if name == "adam":
		return optimizers.Adam(alpha=lr, beta1=momentum)
	raise NotImplementationError()

def decay_learning_rate(opt, factor, final_value):
	if isinstance(opt, optimizers.NesterovAG):
		if opt.lr <= final_value:
			return
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.SGD):
		if opt.lr <= final_value:
			return
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.Adam):
		if opt.alpha <= final_value:
			return
		opt.alpha *= factor
		return
	raise NotImplementationError()

def compute_character_error_rate(r, h):
	if len(r) == 0:
		return len(h)
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

def decay_learning_rate(opt, factor, final_value):
	if isinstance(opt, optimizers.NesterovAG):
		if opt.lr <= final_value:
			return
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.Adam):
		if opt.alpha <= final_value:
			return
		opt.alpha *= factor
		return
	raise NotImplementationError()

def compute_error(model, buckets_indices, buckets_feature, buckets_feature_length, buckets_sentence, buckets_batchsize, BLANK, mean_x_batch, stddev_x_batch, approximate=True):
	errors = []
	xp = model.xp
	for bucket_idx in xrange(len(buckets_indices)):
		data_indices = buckets_indices[bucket_idx]
		batchsize = buckets_batchsize[bucket_idx]
		feature_bucket = buckets_feature[bucket_idx]
		feature_length_bucket = buckets_feature_length[bucket_idx]
		sentence_bucket = buckets_sentence[bucket_idx]

		total_iterations = 1 if approximate else int(math.ceil(len(data_indices) / batchsize))

		if total_iterations == 1 and len(data_indices) < batchsize:
			batchsize = len(data_indices)

		sum_error = 0
		for itr in xrange(1, total_iterations + 1):

			x_batch, x_length_batch, t_batch, t_length_batch = get_minibatch(data_indices, feature_bucket, feature_length_bucket, sentence_bucket, batchsize, BLANK)
			x_batch = (x_batch - mean_x_batch) / stddev_x_batch

			if model.xp is cuda.cupy:
				x_batch = cuda.to_gpu(x_batch.astype(np.float32))
				t_batch = cuda.to_gpu(np.asarray(t_batch).astype(np.int32))
				x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
				t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

			y_batch = model(x_batch, split_into_variables=False)
			y_batch = xp.argmax(y_batch.data, axis=2)

			for argmax_sequence, true_sequence in zip(y_batch, t_batch):
				target_sequence = []
				for token in true_sequence:
					if token == BLANK:
						continue
					target_sequence.append(int(token))
				pred_seqence = []
				prev_token = BLANK
				for token in argmax_sequence:
					if token == BLANK:
						prev_token = BLANK
						continue
					if token == prev_token:
						continue
					pred_seqence.append(int(token))
					prev_token = token
				# if approximate == True:
				# 	print("true:", target_sequence, "pred:", pred_seqence)
				error = compute_character_error_rate(target_sequence, pred_seqence)
				sum_error += error


			sys.stdout.write("\r" + stdout.CLEAR)
			sys.stdout.write("\rComputing error - bucket {}/{} - iteration {}/{}".format(bucket_idx + 1, len(buckets_indices), itr, total_iterations))
			sys.stdout.flush()
			data_indices = np.roll(data_indices, batchsize)

		errors.append(sum_error * 100.0 / batchsize / total_iterations)
	return errors

def main():
	sampling_rate = 16000
	frame_width = 0.032		# sec
	frame_shift = 0.01		# sec
	gpu_ids = [0, 1, 3]		# 複数GPUを使う場合
	num_fft = int(sampling_rate * frame_width)
	num_mel_filters = 40

	chainer.global_config.sampling_rate = sampling_rate
	chainer.global_config.frame_width = frame_width
	chainer.global_config.frame_shift = frame_shift
	chainer.global_config.num_fft = num_fft
	chainer.global_config.num_mel_filters = num_mel_filters
	chainer.global_config.window_func = lambda x:np.hanning(x)
	chainer.global_config.using_delta = True
	chainer.global_config.using_delta_delta = True

	# データの読み込み
	_buckets_feature, _buckets_feature_length, _buckets_sentence, mean_x_batch, stddev_x_batch = load_buckets(args.buckets_limit, args.data_limit)

	# ミニバッチを取れないものは除外
	batchsizes = [64, 64, 32, 32, 32, 24, 24, 24, 16, 16, 16, 4, 4, 4, 4, 4, 4, 4, 4]
	batchsizes = batchsizes[:len(_buckets_feature)]

	buckets_feature = []
	buckets_feature_length = []
	buckets_sentence = []
	buckets_batchsize = []

	dataset_size = 0

	for bucket_idx, (feature_bucket, batchsize) in enumerate(zip(_buckets_feature, batchsizes)):
		if len(feature_bucket) < batchsize:
			continue
		if args.buckets_limit is not None and bucket_idx > args.buckets_limit:
			continue
		buckets_feature.append(_buckets_feature[bucket_idx])
		buckets_feature_length.append(_buckets_feature_length[bucket_idx])
		buckets_sentence.append(_buckets_sentence[bucket_idx])
		buckets_batchsize.append(batchsize)
		dataset_size += len(buckets_feature[-1])
	
	buckets_size = len(buckets_feature)
	vocab, vocab_inv, BLANK = get_vocab()
	vocab_size = len(vocab)
	print_bold("data	#")
	print("audio	{}".format(dataset_size))
	print("vocab	{}".format(vocab_size))

	# 訓練データとテストデータに分ける
	print_bold("bucket	#train	#dev	sec")
	np.random.seed(args.seed)
	buckets_indices_train = []
	buckets_indices_dev = []
	total_train = 0
	total_dev = 0
	for bucket_idx in xrange(buckets_size):
		bucket = buckets_feature[bucket_idx]
		bucket_size = len(bucket)
		num_dev = int(bucket_size * args.dev_split)
		indices = np.arange(0, bucket_size)
		np.random.shuffle(indices)
		indices_dev = indices[:num_dev]
		indices_train = indices[num_dev:]
		buckets_indices_train.append(indices_train)
		buckets_indices_dev.append(indices_dev)
		print("{}	{:>6}	{:>4}	{:>6.3f}".format(bucket_idx + 1, len(indices_train), len(indices_dev), get_duration_seconds(bucket.shape[3])))
		total_train += len(indices_train)
		total_dev += len(indices_dev)
	print("total	{:>6}	{:>4}".format(total_train, total_dev))

	# バケットごとのデータ量の差を学習回数によって補正する
	# データが多いバケットほど多くの学習（ミニバッチのサンプリング）を行う
	batchsizes = batchsizes[:len(buckets_indices_train)]
	required_interations = []
	for bucket, batchsize in zip(buckets_indices_train, batchsizes):
		itr = int(math.ceil(len(bucket) / batchsize))
		required_interations.append(itr)
	total_iterations_train = sum(required_interations)
	buckets_distribution = np.asarray(required_interations, dtype=float) / total_iterations_train

	# モデル
	chainer.global_config.vocab_size = vocab_size
	chainer.global_config.ndim_audio_features = args.ndim_audio_features
	chainer.global_config.ndim_h = args.ndim_h
	chainer.global_config.ndim_dense = args.ndim_dense
	chainer.global_config.kernel_size = (3, 5)
	chainer.global_config.dropout = args.dropout
	chainer.global_config.weightnorm = args.weightnorm
	chainer.global_config.wgain = args.wgain
	chainer.global_config.architecture = args.architecture

	model = load_model(args.model_dir)
	if model is None:
		config = chainer.config
		model = build_model(vocab_size=vocab_size, ndim_audio_features=config.ndim_audio_features, ndim_h=config.ndim_h, ndim_dense=config.ndim_dense,
		 kernel_size=(3, 5), dropout=config.dropout, weightnorm=config.weightnorm, wgain=config.wgain,
		 num_mel_filters=config.num_mel_filters, architecture=config.architecture)






	# model = ZhangModel(vocab_size, 4, 3, 3, 128, ndim_fc=320, nonlinearity="relu", kernel_size=(3, 5), dropout=0, layernorm=True, weightnorm=False, residual=True, wgain=1, num_mel_filters=40)










	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu(args.gpu_device)
	xp = model.xp

	# optimizer
	optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
	final_learning_rate = 1e-4
	total_time = 0

	for epoch in xrange(1, args.total_epoch + 1):
		print_bold("Epoch %d" % epoch)
		start_time = time.time()
		sum_loss = 0
		
		with chainer.using_config("train", True):
			for itr in xrange(1, total_iterations_train + 1):
				bucket_idx = int(np.random.choice(np.arange(len(buckets_indices_train)), size=1, p=buckets_distribution))
				data_indices = buckets_indices_train[bucket_idx]
				batchsize = buckets_batchsize[bucket_idx]
				np.random.shuffle(data_indices)

				feature_bucket = buckets_feature[bucket_idx]
				feature_length_bucket = buckets_feature_length[bucket_idx]
				sentence_bucket = buckets_sentence[bucket_idx]

				x_batch, x_length_batch, t_batch, t_length_batch = get_minibatch(data_indices, feature_bucket, feature_length_bucket, sentence_bucket, batchsize, BLANK)

				# 正規化
				x_batch = (x_batch - mean_x_batch) / stddev_x_batch

				# GPU
				if xp is cupy:
					x_batch = cuda.to_gpu(x_batch.astype(np.float32))
					t_batch = cuda.to_gpu(t_batch.astype(np.int32))
					x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
					t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

				# 誤差の計算
				y_batch = model(x_batch)	# list of variables
				loss = connectionist_temporal_classification(y_batch, t_batch, BLANK, x_length_batch, t_length_batch)

				# NaN
				loss_value = float(loss.data)
				if loss_value == loss_value:
					# 更新
					optimizer.update(lossfun=lambda: loss)
				else:
					print("Encountered NaN when computing loss.")

				sum_loss += loss_value
				sys.stdout.write("\r" + stdout.CLEAR)
				sys.stdout.write("\riteration {}/{}".format(itr, total_iterations_train))
				sys.stdout.flush()

		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()
		save_model(args.model_dir, model)

		# バリデーション
		with chainer.using_config("train", False):
			train_error = compute_error(model, buckets_indices_train, buckets_feature, buckets_feature_length, buckets_sentence, buckets_batchsize, BLANK, mean_x_batch, stddev_x_batch, approximate=True)
			dev_error = compute_error(model, buckets_indices_dev, buckets_feature, buckets_feature_length, buckets_sentence, buckets_batchsize, BLANK, mean_x_batch, stddev_x_batch, approximate=False)

		sys.stdout.write(stdout.MOVE)
		sys.stdout.write(stdout.LEFT)

		# ログ
		elapsed_time = time.time() - start_time
		print("Epoch {} done in {} min - total {} min".format(epoch, int(elapsed_time / 60), int(total_time / 60)))
		sys.stdout.write(stdout.CLEAR)
		print("	loss:", sum_loss / total_iterations_train)
		print("	CER (train):	", train_error)
		print("	CER (dev):	", dev_error)
		print("	lr: {}".format(get_current_learning_rate(optimizer)))
		total_time += elapsed_time

		# 学習率の減衰
		decay_learning_rate(optimizer, args.lr_decay, final_learning_rate)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--total-epoch", "-e", type=int, default=1000)
	parser.add_argument("--grad-clip", "-gc", type=float, default=1) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=1e-5) 
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.001)
	parser.add_argument("--lr-decay", "-decay", type=float, default=0.95)
	parser.add_argument("--momentum", "-mo", type=float, default=0.9)
	parser.add_argument("--optimizer", "-opt", type=str, default="adam")
	
	parser.add_argument("--ndim-audio-features", "-features", type=int, default=3)
	parser.add_argument("--ndim-h", "-dh", type=int, default=320)
	parser.add_argument("--ndim-dense", "-dd", type=int, default=1024)
	parser.add_argument("--wgain", "-w", type=float, default=1)

	parser.add_argument("--nonlinear", type=str, default="relu")
	parser.add_argument("--dropout", "-dropout", type=float, default=0)
	parser.add_argument("--weightnorm", "-weightnorm", default=False, action="store_true")
	parser.add_argument("--architecture", "-arch", type=str, default="zhang")
	
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--interval", type=int, default=100)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--dev-split", "-split", type=float, default=0.05)
	parser.add_argument("--train-filename", "-train", default=None)
	parser.add_argument("--dev-filename", "-dev", default=None)

	parser.add_argument("--buckets-limit", type=int, default=None)
	parser.add_argument("--data-limit", type=int, default=None)
	parser.add_argument("--seed", "-seed", type=int, default=0)
	args = parser.parse_args()

	main()
