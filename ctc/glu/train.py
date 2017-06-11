# coding: utf8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, argparse, time
import chainer
import numpy as np
import chainer.functions as F
from chainer import optimizers, cuda
sys.path.append("../../")
from dataset import load_audio_and_transcription, get_minibatch, get_vocab
from model import ZhangModel, load_model, save_model

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"
	MOVE = "\033[1A"
	LEFT = "\033[G"

def print_bold(str):
	print(stdout.BOLD + str + stdout.END)

# バケットのインデックスを計算
def get_bucket_index(signal):
	return len(signal) // 512 // 16

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

def main(args):
	wav_paths = [
		"/home/stark/sandbox/CSJ/WAV/core/",
	]

	transcription_paths = [
		"/home/stark/sandbox/CSJ_/core/",
	]

	np.random.seed(0)

	sampling_rate = 16000
	frame_width = 0.032		# 秒
	frame_shift = 0.01		# 秒
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

	pair = load_audio_and_transcription(wav_paths, transcription_paths)
	vocab, vocab_inv, ID_PAD, ID_BLANK = get_vocab()
	vocab_size = len(vocab)
	print_bold("data	#")
	print("wav	{}".format(len(pair)))
	print("vocab	{}".format(len(vocab)))

	dataset = []
	max_bucket_index = 0	# バケットの最大個数
	for signal, sentence in pair:
		# 転記、対数メルフィルタバンク出力、Δ、ΔΔの順で並べる
		# データが多いことが想定されるので遅延読み込み
		dataset.append((signal, sentence, None, None, None))
		bucket_index = get_bucket_index(signal)
		if bucket_index > max_bucket_index:
			max_bucket_index = bucket_index

	tmp_buckets = [None] * (max_bucket_index + 1)
	for idx, data in enumerate(dataset):
		signal = data[0]
		bucket_index = get_bucket_index(signal)
		if tmp_buckets[bucket_index] is None:
			tmp_buckets[bucket_index] = []
		tmp_buckets[bucket_index].append(idx)

	buckets_train = []
	buckets_dev = []
	print_bold("bucket	#train	#dev	sec")
	for idx, bucket in enumerate(tmp_buckets):
		if bucket is None:
			continue
		if len(bucket) < args.batchsize:	# ミニバッチサイズより少ない場合はスキップ
			continue
		if args.buckets_slice is not None and idx > args.buckets_slice:
			continue

		# split into train and dev
		bucket = np.asarray(bucket)
		np.random.shuffle(bucket)
		num_dev = int(len(bucket) * args.dev_split)
		bucket_dev = bucket[:num_dev]
		bucket_train = bucket[num_dev:]
		buckets_dev.append(bucket_dev)
		buckets_train.append(bucket_train)

		print("{}	{:>6}	{:>4}	{:>6.3f}".format(idx + 1, len(bucket_train), len(bucket_dev), (idx + 1) * 512 * 16 / sampling_rate))

	print("total {} buckets.".format(len(buckets_train)))

	# バケットごとのデータ量の差を学習回数によって補正する
	# データが多いバケットほど多くの学習（ミニバッチのサンプリング）を行う
	required_interations = []
	for data in buckets_train:
		itr = len(data) // args.batchsize + 1
		required_interations.append(itr)
	total_iterations_train = sum(required_interations)
	buckets_distribution = np.asarray(required_interations, dtype=float) / total_iterations_train

	# numpy配列に変換
	for idx, bucket in enumerate(buckets_train):
		bucket = np.asarray(bucket)
		np.random.shuffle(bucket)
		buckets_train[idx] = bucket

	# モデル
	model = load_model(args.model_dir)
	if model is None:
		model = ZhangModel(vocab_size, args.num_blocks, args.num_layers_per_block, args.num_fc_layers, args.ndim_audio_features, args.ndim_h, dropout=args.dropout, weightnorm=args.weightnorm, wgain=args.wgain, ignore_label=ID_PAD)
	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	# optimizer
	optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
	final_learning_rate = 1e-4
	total_time = 0

	running_mean = 0
	running_stddev = 0
	running_z = 0

	for epoch in xrange(1, args.epoch + 1):
		print("Epoch", epoch)
		start_time = time.time()
		sum_loss = 0
		
		with chainer.using_config("train", True):

			for itr in xrange(1, total_iterations_train + 1):
				bucket_idx = int(np.random.choice(np.arange(len(buckets_train)), size=1, p=buckets_distribution))
				bucket = buckets_train[bucket_idx]
				x_batch, x_length_batch, t_batch, t_length_batch = get_minibatch(bucket, dataset, args.batchsize, ID_PAD)
				feature_dim = x_batch.shape[1]

				# 平均と分散を計算
				# 本来は学習前にデータ全体の平均・分散を計算すべきだが、データ拡大を用いるため、逐一更新していくことにする
				mean_x_batch = np.mean(x_batch, axis=(0, 3)).reshape((1, feature_dim, -1, 1))
				stddev_x_batch = np.std(x_batch, axis=(0, 3)).reshape((1, feature_dim, -1, 1))
				running_mean = running_mean * (running_z / (running_z + 1)) + mean_x_batch / (running_z + 1)	# 正規化定数が+1されることに注意
				running_stddev = running_stddev * (running_z / (running_z + 1)) + stddev_x_batch / (running_z + 1)		# 正規化定数が+1されることに注意
				running_z += 1

				# 正規化
				x_batch = (x_batch - running_mean) / running_stddev

				# GPU
				if model.xp is cuda.cupy:
					x_batch = cuda.to_gpu(x_batch.astype(np.float32))
					t_batch = cuda.to_gpu(np.asarray(t_batch).astype(np.int32))
					x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
					t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

				# 誤差の計算
				y_batch = model(x_batch)	# list of variables
				loss = F.connectionist_temporal_classification(y_batch, t_batch, ID_BLANK, x_length_batch, t_length_batch)

				# 更新
				model.cleargrads()
				loss.backward()
				optimizer.update()

				buckets_train[bucket_idx] = np.roll(bucket, args.batchsize)	# ずらす

				loss = float(loss.data)
				if loss != loss:
					raise Exception("loss is NaN")
				sum_loss += loss

				sys.stdout.write("\r" + stdout.CLEAR)
				sys.stdout.write("\riteration {}/{}".format(itr, total_iterations_train))
				sys.stdout.flush()

		# 再シャッフル
		for bucket in buckets_train:
			np.random.shuffle(bucket)

		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()

		# バリデーション
		with chainer.using_config("train", False):
			train_error = []
			dev_error = []

			# train
			for bucket_idx in xrange(len(buckets_train)):
				bucket = buckets_train[bucket_idx]
				sum_error = 0

				x_batch, x_length_batch, t_batch, t_length_batch = get_minibatch(bucket, dataset, args.batchsize_dev, ID_PAD)
				x_batch = (x_batch - running_mean) / running_stddev

				if model.xp is cuda.cupy:
					x_batch = cuda.to_gpu(x_batch.astype(np.float32))
					t_batch = cuda.to_gpu(np.asarray(t_batch).astype(np.int32))
					x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
					t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

				y_batch = model(x_batch, split_into_variables=False).data
				T = y_batch.shape[1]
				xp = model.xp

				for batch_idx in xrange(len(y_batch)):
					y_sequence = y_batch[batch_idx]
					t_sequence = t_batch[batch_idx]

					pred_ids = []
					for t in xrange(T):
						prob = y_sequence[t]
						assert prob.size == vocab_size
						char_id = int(xp.argmax(prob))
						if char_id == ID_PAD:
							continue
						if char_id == ID_BLANK:
							continue
						pred_ids.append(char_id)

					target_ids = []
					for t in xrange(t_sequence.size):
						char_id = int(t_sequence[t])
						if char_id == ID_PAD:
							continue
						if char_id == ID_BLANK:
							continue
						target_ids.append(char_id)

					error = compute_character_error_rate(target_ids, pred_ids)
					sum_error += error

				train_error.append(sum_error * 100.0 / args.batchsize)

			# dev
			for bucket_idx in xrange(len(buckets_dev)):
				bucket = buckets_dev[bucket_idx]
				total_iterations_dev = int(len(bucket) // args.batchsize_dev)
				sum_error = 0

				for itr in xrange(total_iterations_dev):
					x_batch, x_length_batch, t_batch, t_length_batch = get_minibatch(bucket, dataset, args.batchsize_dev, ID_PAD)
					x_batch = (x_batch - running_mean) / running_stddev

					if model.xp is cuda.cupy:
						x_batch = cuda.to_gpu(x_batch.astype(np.float32))
						t_batch = cuda.to_gpu(np.asarray(t_batch).astype(np.int32))
						x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
						t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

					y_batch = model(x_batch, split_into_variables=False).data
					T = y_batch.shape[1]
					xp = model.xp

					for batch_idx in xrange(len(y_batch)):
						y_sequence = y_batch[batch_idx]
						t_sequence = t_batch[batch_idx]

						pred_ids = []
						for t in xrange(T):
							prob = y_sequence[t]
							assert prob.size == vocab_size
							char_id = int(xp.argmax(prob))
							if char_id == ID_PAD:
								continue
							if char_id == ID_BLANK:
								continue
							pred_ids.append(char_id)

						target_ids = []
						for t in xrange(t_sequence.size):
							char_id = int(t_sequence[t])
							if char_id == ID_PAD:
								continue
							if char_id == ID_BLANK:
								continue
							target_ids.append(char_id)

						error = compute_character_error_rate(target_ids, pred_ids)
						sum_error += error

					sys.stdout.write("\r" + stdout.CLEAR)
					sys.stdout.write("\rComputing validation error - bucket {}/{} - iteration {}/{}".format(bucket_idx + 1, len(buckets_dev), itr, total_iterations_dev))
					sys.stdout.flush()

					buckets_dev[bucket_idx] = np.roll(bucket, args.batchsize_dev)

				dev_error.append(sum_error * 100.0 / args.batchsize / total_iterations_dev)

			for bucket in buckets_dev:
				np.random.shuffle(bucket)

		sys.stdout.write(stdout.MOVE)
		sys.stdout.write(stdout.LEFT)

		elapsed_time = time.time() - start_time
		print("Epoch {} done in {} min".format(epoch, int(elapsed_time / 60)))
		sys.stdout.write(stdout.CLEAR)
		print("	loss:", sum_loss / total_iterations_train)
		print("	CER:", train_error, dev_error)
		total_time += elapsed_time


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=8)
	parser.add_argument("--batchsize-dev", "-bd", type=int, default=8)
	parser.add_argument("--epoch", "-e", type=int, default=1000)
	parser.add_argument("--grad-clip", "-gc", type=float, default=1) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=1e-5) 
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.001)
	parser.add_argument("--lr-decay", "-decay", type=float, default=0.95)
	parser.add_argument("--momentum", "-mo", type=float, default=0.9)
	parser.add_argument("--optimizer", "-opt", type=str, default="adam")
	
	parser.add_argument("--ndim-audio-features", "-features", type=int, default=3)
	parser.add_argument("--ndim-h", "-nh", type=int, default=320)
	parser.add_argument("--num-layers-per-block", "-layers", type=int, default=2)
	parser.add_argument("--num-blocks", "-blocks", type=int, default=1)
	parser.add_argument("--num-fc-layers", "-fc", type=int, default=1)
	parser.add_argument("--wgain", "-w", type=float, default=1)

	parser.add_argument("--dropout", "-dropout", type=float, default=0)
	parser.add_argument("--weightnorm", "-weightnorm", default=False, action="store_true")
	
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--interval", type=int, default=100)
	parser.add_argument("--buckets-slice", type=int, default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--dev-split", "-split", type=float, default=0.05)
	parser.add_argument("--train-filename", "-train", default=None)
	parser.add_argument("--dev-filename", "-dev", default=None)
	args = parser.parse_args()

	main(args)