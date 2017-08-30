# coding: utf-8
from __future__ import division
from __future__ import print_function
import os, codecs, re, sys, math, chainer, pickle, acoustics, argparse
import numpy as np
import scipy.io.wavfile as wavfile
from chainer import cuda
sys.path.append("../")
import config, fft
from util import stdout, printb, printr
from dataset import wav_path_list, transcription_path_list, cache_path, get_bucket_index, generate_signal_transcription_pairs

wav_path_list = [
	"/home/aibo/sandbox/CSJ/WAV/core",
	"/home/aibo/sandbox/CSJ/WAV/noncore",
]
# 変換済みの書き起こしデータ
# https://github.com/musyoku/csj-preprocesser
transcription_path_list = [
	"/home/aibo/sandbox/CSJ_/core",
	"/home/aibo/sandbox/CSJ_/noncore",
]

def normalize_feature(array):
	mean = np.mean(array)
	stddev = np.std(array)
	array = (array - mean) / stddev
	return array

def generate_buckets(wav_paths, transcription_paths, cache_path, buckets_limit, data_limit, num_signals_per_file=1000):
	assert len(wav_paths) > 0
	assert len(transcription_paths) > 0

	config = chainer.config
	buckets_signal = []
	buckets_sentence = []
	buckets_file_indices = []
	max_sentence_length = 0
	max_logmel_length = 0
	current_num_data = 0
	data_limit_exceeded = False
	total_min = 0
	statistics_denominator = 0
	feature_mean = 0
	feature_std = 0

	def append_bucket(buckets, bucket_idx, data):
		if len(buckets) <= bucket_idx:
			while len(buckets) <= bucket_idx:
				buckets.append([])
		buckets[bucket_idx].append(data)

	def add_to_bukcet(signal, sentence):
		bucket_idx = get_bucket_index(signal, config.sampling_rate, config.bucket_split_sec)
		# add signal
		append_bucket(buckets_signal, bucket_idx, signal)
		# add sentence
		append_bucket(buckets_sentence, bucket_idx, sentence)
		# add file index
		if len(buckets_file_indices) <= bucket_idx:
			while len(buckets_file_indices) <= bucket_idx:
				buckets_file_indices.append(0)
		# check
		if len(buckets_signal[bucket_idx]) >= num_signals_per_file:
			return True, bucket_idx
		return False, bucket_idx

	def save_bucket(bucket_idx):
		if buckets_limit is not None and bucket_idx > buckets_limit:
			return False
		file_index = buckets_file_indices[bucket_idx]
		num_signals = len(buckets_signal[bucket_idx])
		assert num_signals > 0

		with open (os.path.join(cache_path, "signal", "{}_{}_{}.bucket".format(bucket_idx, file_index, num_signals)), "wb") as f:
			pickle.dump(buckets_signal[bucket_idx], f)

		num_sentences = len(buckets_sentence[bucket_idx])
		assert num_signals == num_sentences
		with open (os.path.join(cache_path, "sentence", "{}_{}_{}.bucket".format(bucket_idx, file_index, num_sentences)), "wb") as f:
			pickle.dump(buckets_sentence[bucket_idx], f)
		buckets_signal[bucket_idx] = []
		buckets_sentence[bucket_idx] = []
		buckets_file_indices[bucket_idx] += 1
		return True

	def compute_mean_and_std(bucket_idx, apply_cmn=False):
		num_signals = len(buckets_signal[bucket_idx])
		assert num_signals > 0
		mean = 0
		std = 0

		for signal in buckets_signal[bucket_idx]:
			spec = fft.get_specgram(signal, config.sampling_rate, nfft=config.num_fft, winlen=config.frame_width, winstep=config.frame_shift, winfunc=config.window_func)

			# ケプストラム平均正規化
			if apply_cmn:
				log_spec = np.log(spec + 1e-16)
				spec = np.exp(log_spec - np.mean(log_spec, axis=0))

			logmel = fft.compute_logmel(spec, config.sampling_rate, nfft=config.num_fft, winlen=config.frame_width, winstep=config.frame_shift, nfilt=config.num_mel_filters, winfunc=config.window_func)
			logmel, delta, delta_delta = fft.compute_deltas(logmel)

			assert logmel.shape[0] > 0
			assert delta.shape[0] > 0
			assert delta_delta.shape[0] > 0

			# 発話ごとに平均0、分散1にする
			# 発話ごとの場合データが少ないので全軸で取らないとノイズが増大する
			# if apply_cmn:
			# 	logmel = normalize_feature(logmel)
			# 	delta = normalize_feature(delta)
			# 	delta_delta = normalize_feature(delta_delta)

			# 目視チェック
			sys.path.append(os.path.join("..", "visual"))
			from specgram import _plot_features
			_plot_features("/home/aibo/sandbox/plot", signal, config.sampling_rate, logmel, delta, delta_delta, spec, 
				str(np.random.randint(0, 5000)) + (".norm." if apply_cmn else "") + ".png")

			logmel = logmel.T
			delta = delta.T
			delta_delta = delta_delta.T

			if logmel.shape[1] == 0:
				continue
			div = 1
			feature = logmel[:, None, :]
			if config.using_delta:
				feature = np.concatenate((feature, delta[:, None, :]), axis=1)
				div += 1
			if config.using_delta_delta:
				feature = np.concatenate((feature, delta_delta[:, None, :]), axis=1)
				div += 1

			# 目視チェック
			# sys.path.append("../visual")
			# from specgram import _plot_features
			# _plot_features("/home/aibo/sandbox/plot", signal, config.sampling_rate, feature[:, 0].T, feature[:, 1].T, feature[:, 2].T, spec, str(np.random.randint(0, 5000)) + ".png")

			_mean = np.mean(feature, axis=2, keepdims=True) / num_signals
			_std = np.std(feature, axis=2, keepdims=True) / num_signals
			assert np.isnan(np.sum(_mean)) == False
			assert np.isnan(np.sum(_std)) == False
			mean += _mean
			std += _std

		return mean, std

	for wav_dir, trn_dir in zip(wav_paths, transcription_paths):
		wav_fs = os.listdir(wav_dir)
		trn_fs = os.listdir(trn_dir)
		wav_ids = set()
		trn_ids = set()
		for filename in wav_fs:
			data_id = re.sub(r"\..+$", "", filename)
			wav_ids.add(data_id)
		for filename in trn_fs:
			data_id = re.sub(r"\..+$", "", filename)
			trn_ids.add(data_id)

		if data_limit_exceeded:
			break

		for data_idx, wav_id in enumerate(sorted(wav_ids)):

			if data_limit_exceeded:
				break

			if wav_id not in trn_ids:
				sys.stdout.write("\r")
				sys.stdout.write(stdout.CLEAR)
				print("%s.trn not found" % wav_id)
				continue

			wav_filename = "%s.wav" % wav_id
			trn_filename = "%s.trn" % wav_id

			# wavの読み込み
			try:
				sampling_rate, audio = wavfile.read(os.path.join(wav_dir, wav_filename))
			except KeyboardInterrupt:
				exit()
			except Exception as e:
				printr("Failed to read {} ({})".format(wav_filename, str(e)))
				continue

			duration = audio.size / sampling_rate / 60
			total_min += duration

			printr("Loading {} ({}/{}) ... shape={}, rate={}, min={}, #buckets={}, #data={}".format(wav_filename, data_idx + 1, len(wav_fs), audio.shape, sampling_rate, int(duration), len(buckets_signal), current_num_data))

			# 転記の読み込みと音声の切り出し
			signal_transcription_pairs = generate_signal_transcription_pairs(os.path.join(trn_dir, trn_filename), audio, sampling_rate)

			for idx, (signal_sequence, sentence) in enumerate(signal_transcription_pairs):
				# データを確認する場合は書き出し
				# wavfile.write("/home/stark/sandbox/debug/{}.wav".format(sentence), config.sampling_rate, signal_sequence)
				
				write_to_file, bucket_idx = add_to_bukcet(signal_sequence, sentence)
				if write_to_file:
					printr("Computing mean and std of bucket {} ...".format(bucket_idx))

					mean, std = compute_mean_and_std(bucket_idx, args.apply_cmn)
					feature_mean += mean
					feature_std += std
					statistics_denominator += 1

					printr("Writing bucket {} ...".format(bucket_idx))

					save_bucket(bucket_idx)
					current_num_data += num_signals_per_file

					if data_limit is not None and current_num_data >= data_limit:
						data_limit_exceeded = True
						break

	printr("")
	if data_limit_exceeded == False:
		for bucket_idx, bucket in enumerate(buckets_signal):
			if len(bucket) > 0:
				mean, std = compute_mean_and_std(bucket_idx, args.apply_cmn)
				feature_mean += mean
				feature_std += std
				statistics_denominator += 1
				if save_bucket(bucket_idx) == False:
					num_unsaved_data = len(buckets_signal[bucket_idx])
					print("bucket {} skipped. (#data={})".format(bucket_idx, num_unsaved_data))
					current_num_data -= num_unsaved_data
	print("total: {} hour - {} data".format(int(total_min / 60), current_num_data))

	feature_mean /= statistics_denominator
	feature_std /= statistics_denominator
	np.save(os.path.join(cache_path, "mean.npy"), feature_mean.swapaxes(0, 1))
	np.save(os.path.join(cache_path, "std.npy"), feature_std.swapaxes(0, 1))

def mkdir(d):
	try:
		os.mkdir(d)
	except:
		pass

if __name__ == "__main__":
	mkdir(cache_path)
	mkdir(os.path.join(cache_path, "signal"))
	mkdir(os.path.join(cache_path, "sentence"))
	np.random.seed(0)

	parser = argparse.ArgumentParser()
	parser.add_argument("--apply-cmn", "-cmn", default=False, action="store_true")
	args = parser.parse_args()

	# すべての.wavを読み込み、一定の長さごとに保存
	generate_buckets(wav_path_list, transcription_path_list, cache_path, buckets_limit=20, data_limit=None, num_signals_per_file=500)