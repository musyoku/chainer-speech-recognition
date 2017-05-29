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

	bucketset = []
	min_num_data = 0
	print_bold("bucket	#data	sec")
	for idx, bucket in enumerate(tmp_buckets):
		if bucket is None:
			continue
		if len(bucket) < args.batchsize:	# ミニバッチサイズより少ない場合はスキップ
			continue
		if args.buckets_slice is not None and idx > args.buckets_slice:
			continue
		print("{}	{:>5}	{:>6.3f}".format(idx + 1, len(bucket), (idx + 1) * 512 * 16 / sampling_rate))
		bucketset.append(bucket)
		if min_num_data == 0 or len(bucket) < min_num_data:
			min_num_data = len(bucket)
	print("total {} buckets.".format(len(bucketset)))

	# バケットごとのデータ量の差を学習回数によって補正する
	# データが多いバケットほど多くの学習（ミニバッチのサンプリング）を行う
	required_interations = []
	for data in bucketset:
		itr = len(data) // args.batchsize + 1
		required_interations.append(itr)
	total_iterations = sum(required_interations)
	buckets_distribution = np.asarray(required_interations, dtype=float) / total_iterations

	# numpy配列に変換
	for idx, bucket in enumerate(bucketset):
		bucket = np.asarray(bucket)
		np.random.shuffle(bucket)
		bucketset[idx] = bucket
	max_epoch = 2

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
	running_var = 0
	running_z = 0

	for epoch in xrange(1, max_epoch + 1):
		print("Epoch", epoch)
		start_time = time.time()
		
		for itr in xrange(1, total_iterations + 1):
			with chainer.using_config("debug", True):
				bucket_idx = int(np.random.choice(np.arange(len(bucketset)), size=1, p=buckets_distribution))
				bucket = bucketset[bucket_idx]
				x_batch, x_length_batch, t_batch, t_length_batch = get_minibatch(bucket, dataset, args.batchsize, ID_PAD)
				feature_dim = x_batch.shape[1]

				# 平均と分散を計算
				# 本来は学習前にデータ全体の平均・分散を計算すべきだが、データ拡大を用いるため、逐一更新していくことにする
				mean_x_batch = np.mean(x_batch, axis=(0, 3)).reshape((1, feature_dim, -1, 1))
				var_x_batch = np.var(x_batch, axis=(0, 3)).reshape((1, feature_dim, -1, 1))
				running_mean = running_mean * (running_z / (running_z + 1)) + mean_x_batch / (running_z + 1)	# 正規化定数が+1されることに注意
				running_var = running_var * (running_z / (running_z + 1)) + var_x_batch / (running_z + 1)		# 正規化定数が+1されることに注意
				running_z += 1

				# 正規化
				x_batch = (x_batch - running_mean) / running_var

				# GPU
				if model.xp is cuda.cupy:
					x_batch = cuda.to_gpu(x_batch)
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

				sys.stdout.write("\r" + stdout.CLEAR)
				sys.stdout.write("\riteration {}/{}".format(itr, total_iterations))
				sys.stdout.flush()

				bucketset[bucket_idx] = np.roll(bucket, args.batchsize)	# ずらす

			# 再シャッフル
			for bucket in bucketset:
				np.random.shuffle(bucket)

			sys.stdout.write("\r" + stdout.CLEAR)
			sys.stdout.flush()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=8)
	parser.add_argument("--epoch", "-e", type=int, default=1000)
	parser.add_argument("--grad-clip", "-gc", type=float, default=1) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=2e-5) 
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.1)
	parser.add_argument("--lr-decay", "-decay", type=float, default=0.95)
	parser.add_argument("--momentum", "-mo", type=float, default=0.99)
	parser.add_argument("--optimizer", "-opt", type=str, default="nesterov")
	
	parser.add_argument("--ndim-audio-features", "-features", type=int, default=3)
	parser.add_argument("--ndim-h", "-nh", type=int, default=640)
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
	parser.add_argument("--train-filename", "-train", default=None)
	parser.add_argument("--dev-filename", "-dev", default=None)
	args = parser.parse_args()

	main(args)