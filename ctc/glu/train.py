# coding: utf8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, argparse, time
import chainer
import numpy as np
sys.path.append("../../")
from dataset import load_audio_and_transcription, get_minibatch
from model import RNNModel, load_model, save_model

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

# バケットのインデックスを計算
def get_bucket_index(signal):
	return len(signal) // 512 // 16

def main(args):
	wav_paths = [
		"/home/aibo/sandbox/CSJ/WAV/core/",
	]

	transcription_paths = [
		"/home/aibo/sandbox/CSJ_/core/",
	]

	np.random.seed(0)

	sampling_rate = 16000
	frame_width = 0.032		# 秒
	frame_shift = 0.01		# 秒
	batchsize = 32
	gpu_ids = [0, 1, 3]		# 複数GPUを使う場合
	num_fft = int(sampling_rate * frame_width)
	num_mel_filters = 40

	chainer.global_config.sampling_rate = sampling_rate
	chainer.global_config.frame_width = frame_width
	chainer.global_config.frame_shift = frame_shift
	chainer.global_config.batchsize = batchsize
	chainer.global_config.num_fft = num_fft
	chainer.global_config.num_mel_filters = num_mel_filters
	chainer.config.window_func = lambda x:np.hanning(x)
	chainer.config.using_delta = True
	chainer.config.using_delta_delta = True

	pair = load_audio_and_transcription(wav_paths, transcription_paths)

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
	print("bucket	#data	sec")
	for idx, bucket in enumerate(tmp_buckets):
		if bucket is None:
			continue
		if len(bucket) < batchsize:	# ミニバッチサイズより少ない場合はスキップ
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

	running_mean = 0
	running_var = 0
	running_z = 0

	for epoch in xrange(max_epoch):
		print("Epoch", epoch)
		start_time = time.time()
		
		with chainer.using_config("train", True):
		
			for itr in xrange(1, total_iterations + 1):
				bucket_idx = int(np.random.choice(np.arange(len(bucketset)), size=1, p=buckets_distribution))
				bucket = bucketset[bucket_idx]
				x_batch = get_minibatch(bucket, dataset, batchsize)
				feature_dim = x_batch.shape[1]

				# 平均と分散を計算
				# 本来は学習前にデータ全体の平均・分散を計算すべきだが、データ拡大を用いるため、逐一更新していくことにする
				mean_x_batch = np.mean(x_batch, axis=(0, 3)).reshape((1, feature_dim, -1, 1))
				var_x_batch = np.var(x_batch, axis=(0, 3)).reshape((1, feature_dim, -1, 1))
				running_mean = running_mean * (running_z / (running_z + 1)) + mean_x_batch / (running_z + 1)
				running_var = running_var * (running_z / (running_z + 1)) + var_x_batch / (running_z + 1)
				running_z += 1

				# 正規化
				x_batch = (x_batch - running_mean) / running_var

				sys.stdout.write("\r" + stdout.CLEAR)
				sys.stdout.write("\riteration {}/{}".format(itr, total_iterations))
				sys.stdout.flush()

				bucketset[bucket_idx] = np.roll(bucket, batchsize)	# ずらす

			# 再シャッフル
			for bucket in bucketset:
				np.random.shuffle(bucket)

			sys.stdout.write("\r" + stdout.CLEAR)
			sys.stdout.flush()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=24)
	parser.add_argument("--model-dir", "-model", type=str, default="model")
	args = parser.parse_args()
	main(args)