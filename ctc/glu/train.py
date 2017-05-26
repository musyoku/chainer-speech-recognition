# coding: utf8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, argparse
import chainer
import numpy as np
sys.path.append("../../")
from dataset import load_audio_and_transcription, get_minibatch

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
	gpu_ids = [0, 1, 3]
	num_fft = int(sampling_rate * frame_width)
	num_mel_filter = 40

	chainer.global_config.sampling_rate = sampling_rate
	chainer.global_config.frame_width = frame_width
	chainer.global_config.frame_shift = frame_shift
	chainer.global_config.batchsize = batchsize
	chainer.global_config.num_fft = num_fft
	chainer.global_config.num_mel_filter = num_mel_filter

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
	repeats = [0] * len(bucketset)
	for idx, bucket in enumerate(bucketset):
		repeats[idx] = len(bucket) // min_num_data
	print(repeats)

	# numpy配列に変換
	for idx, bucket in enumerate(bucketset):
		bucket = np.asarray(bucket)
		np.random.shuffle(bucket)
		bucketset[idx] = bucket
	max_epoch = 2

	running_mean = 0
	running_var = 0

	with chainer.using_config("train", True):
		for epoch in xrange(max_epoch):
			for bucket_id, bucket in enumerate(bucketset):

				repeat = repeats[bucket_id]

				for itr in xrange(repeat):

					sys.stdout.write("\r" + stdout.CLEAR)
					sys.stdout.write("\rbucket {}/{} - iteration {}/{}".format(bucket_id + 1, len(bucketset), itr + 1, repeat))
					sys.stdout.flush()

					x_batch = get_minibatch(bucket, dataset, batchsize)

					# 平均と分散を計算
					mean_x_batch = np.expand_dims(np.expand_dims(np.mean(x_batch, axis=(0, 3)), axis=0), axis=3)
					var_x_batch = np.expand_dims(np.expand_dims(np.var(x_batch, axis=(0, 3)), axis=0), axis=3)
					running_mean = (running_mean + mean_x_batch) / 2
					running_var = (running_var + var_x_batch) / 2

					# 正規化
					x_batch = (x_batch - running_mean) / running_var

					# 後処理
					bucket = np.roll(bucket, batchsize)	# ずらす

				bucketset[bucket_id] = bucket

			# 再シャッフル
			for bucket_id, bucket in enumerate(bucketset):
				np.random.shuffle(bucket)

			sys.stdout.write("\r" + stdout.CLEAR)
			sys.stdout.flush()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-dir", "-model", type=str, default="model")
	args = parser.parse_args()
	main(args)