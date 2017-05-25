# coding: utf8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import sys, argparse
sys.path.append("../../")
from dataset import load_audio_and_transcription

# バケットのインデックスを計算
def get_bucket_index(signal):
	return len(signal) // 512 // 16

def main(args):
	wav_paths = [
		"/home/stark/sandbox/CSJ/WAV/core/",
	]

	transcription_paths = [
		"/home/stark/sandbox/CSJ_/core/",
	]

	sampling_rate = 16000
	frame_width = 0.032		# 秒
	frame_shift = 0.01		# 秒
	batchsize = 32
	nfft = int(sampling_rate * frame_width)

	pair = load_audio_and_transcription(wav_paths, transcription_paths)

	dataset = []
	max_bucket_index = 0	# バケットの最大個数
	for signal, sentence in pair:
		# 転記、対数メルフィルタバンク出力、Δ、ΔΔの順で並べる
		# データが多いことが想定されるので適宜読み込むようにする
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

	buckets = []
	print("bucket	#data	sec")
	for idx, bucket in enumerate(tmp_buckets):
		if bucket is None:
			continue
		if len(bucket) < batchsize:	# ミニバッチサイズより少ない場合はスキップ
			continue
		print("{}	{:>5}	{:>6.3f}".format(idx + 1, len(bucket), (idx + 1) * 512 * 16 / sampling_rate))
		buckets.append(bucket)
	print("total {} buckets.".format(len(buckets)))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-dir", "-model", type=str, default="model")
	args = parser.parse_args()
	main(args)