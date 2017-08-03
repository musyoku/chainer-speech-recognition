# coding: utf-8
from __future__ import division
from __future__ import print_function
import os, codecs, re, sys, math, chainer, pickle, acoustics
import numpy as np
import scipy.io.wavfile as wavfile
from chainer import cuda
sys.path.append("../")
import config, fft
from vocab import load_all_tokens, convert_sentence_to_phoneme_sequence, convert_phoneme_sequence_to_triphone_sequence, reduce_triphone, triphonize
from util import stdout, print_bold
from dataset import wav_path_list, transcription_path_list, cache_path

def get_token_id(triphone):
	L, X, R = triphone
	token = triphonize(L, X, R)
	if token in DICTIONARY_TOKEN_ID:
		return DICTIONARY_TOKEN_ID[token]
	return -1

def convert_triphone_to_id(triphone):
	L, X, R = triphone
	L, X, R = reduce_triphone(L, X, R)
	token_id = get_token_id((L, X, R))
	if token_id > 0:
		return token_id
	if L and R:
		token_id = get_token_id((None, X, R))
		if token_id > 0:
			return token_id
	token_id = get_token_id((None, X, None))
	if token_id > 0:
		return token_id
	raise Exception(L, X, R)

def get_bucket_idx(signal, sampling_rate=16000, split_sec=0.5):
	divider = sampling_rate * split_sec
	return int(len(signal) // divider)

def generate_signal_transcription_pairs(trn_path, audio, sampling_rate):
	batch = []
	with codecs.open(trn_path, "r", "utf-8") as f:
		for data in f:
			period_str, channel, sentence = data.split(":")
			sentence = sentence.strip()
			period = period_str.split("-")
			start_sec, end_sec = float(period[0]), float(period[1])
			start_frame = int(start_sec * sampling_rate)
			end_frame = int(end_sec * sampling_rate)

			assert start_frame <= len(audio)
			assert end_frame <= len(audio)

			signal = audio[start_frame:end_frame]

			assert len(signal) == end_frame - start_frame
			if len(signal) < config.num_fft * 3:
				print("\r{}{} skipped. (length={})".format(stdout.CLEAR, sentence, len(signal)))
				continue

			# channelに従って選択
			if signal.ndim == 2:
				if channel == "S":	# 両方に含まれる場合はL
					signal = signal[:, 0]
				elif channel == "L":
					signal = signal[:, 0]
				elif channel == "R":
					signal = signal[:, 1]
				else:
					raise Exception()

			# 音素列をIDに変換
			id_sequence = []
			phoneme_sequence = convert_sentence_to_phoneme_sequence(sentence)
			triphone_sequence = convert_phoneme_sequence_to_triphone_sequence(phoneme_sequence, convert_to_str=False)

			for triphone in triphone_sequence:
				token_id = convert_triphone_to_id(triphone)
				id_sequence.append(token_id)

			batch.append((signal, id_sequence, sentence))
	return batch

def normalize_feature(array):
	mean = np.mean(array)
	stddev = np.std(array)
	array = (array - mean) / stddev
	return array

def generate_buckets(wav_paths, transcription_paths, cache_path, buckets_limit, data_limit, num_signals_per_file=1000):
	assert len(wav_paths) > 0
	assert len(transcription_paths) > 0

	config = chainer.config
	dataset = []
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
		bucket_idx = get_bucket_idx(signal, config.sampling_rate, config.bucket_split_sec)
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

	def compute_mean_and_std(bucket_idx, pre_normalization=True):
		num_signals = len(buckets_signal[bucket_idx])
		assert num_signals > 0
		mean = 0
		std = 0

		for signal in buckets_signal[bucket_idx]:
			spec = fft.get_specgram(signal, config.sampling_rate, nfft=config.num_fft, winlen=config.frame_width, winstep=config.frame_shift, winfunc=config.window_func)

			# ケプストラム平均正規化
			if pre_normalization:
				spec = np.exp(np.log(spec) - np.mean(np.log(spec), axis=0))

			logmel = fft.compute_logmel(spec, config.sampling_rate, nfft=config.num_fft, winlen=config.frame_width, winstep=config.frame_shift, nfilt=config.num_mel_filters, winfunc=config.window_func)
			logmel, delta, delta_delta = fft.compute_deltas(logmel)

			assert logmel.shape[0] > 0
			assert delta.shape[0] > 0
			assert delta_delta.shape[0] > 0

			# 発話ごとに平均0、分散1にする
			# 発話ごとの場合データが少ないので全軸で取らないとノイズが増大する
			if pre_normalization:
				logmel = normalize_feature(logmel)
				delta = normalize_feature(delta)
				delta_delta = normalize_feature(delta_delta)

			# 目視チェック
			# sys.path.append("../visual")
			# from specgram import _plot_features
			# _plot_features("/home/aibo/sandbox/plot", signal, config.sampling_rate, logmel, delta, delta_delta, spec, str(np.random.randint(0, 5000)) + ".png")

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

			mean += np.mean(feature, axis=2, keepdims=True) / num_signals
			std += np.std(feature, axis=2, keepdims=True) / num_signals

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

		for data_idx, wav_id in enumerate(wav_ids):

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
				sys.stdout.write("\r")
				sys.stdout.write(stdout.CLEAR)
				print("Failed to read {} ({})".format(wav_filename, str(e)))
				continue

			duration = audio.size / sampling_rate / 60
			total_min += duration

			sys.stdout.write("\r")
			sys.stdout.write(stdout.CLEAR)
			sys.stdout.write("Loading {} ({}/{}) ... shape={}, rate={}, min={}, #buckets={}, #data={}".format(wav_filename, data_idx + 1, len(wav_fs), audio.shape, sampling_rate, int(duration), len(buckets_signal), current_num_data))
			sys.stdout.flush()

			# 転記の読み込みと音声の切り出し
			signal_transcription_pairs = generate_signal_transcription_pairs(os.path.join(trn_dir, trn_filename), audio, sampling_rate)

			for idx, (signal_sequence, id_sequence, sentence) in enumerate(signal_transcription_pairs):
				# データを確認する場合は書き出し
				# wavfile.write("/home/stark/sandbox/debug/{}.wav".format(sentence), config.sampling_rate, signal_sequence)
				
				write_to_file, bucket_idx = add_to_bukcet(signal_sequence, id_sequence)
				if write_to_file:
					sys.stdout.write("\r")
					sys.stdout.write(stdout.CLEAR)
					sys.stdout.write("Computing mean and std of bucket {} ...".format(bucket_idx))
					sys.stdout.flush()

					mean, std = compute_mean_and_std(bucket_idx)
					feature_mean += mean
					feature_std += std
					statistics_denominator += 1

					sys.stdout.write("\r")
					sys.stdout.write(stdout.CLEAR)
					sys.stdout.write("Writing bucket {} ...".format(bucket_idx))
					sys.stdout.flush()

					save_bucket(bucket_idx)
					current_num_data += num_signals_per_file

					if data_limit is not None and current_num_data >= data_limit:
						data_limit_exceeded = True
						break

	sys.stdout.write("\r")
	sys.stdout.write(stdout.CLEAR)
	if data_limit_exceeded == False:
		for bucket_idx, bucket in enumerate(buckets_signal):
			if len(bucket) > 0:
				mean, std = compute_mean_and_std(bucket_idx)
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

	triphone_list_path = "../triphone.list"
	assert os.path.isfile(triphone_list_path)
	DICTIONARY_TOKEN_ID, DICTIONARY_ID_TOKEN, BLANK = load_all_tokens(triphone_list_path)

	# すべての.wavを読み込み、一定の長さごとに保存
	generate_buckets(wav_path_list, transcription_path_list, cache_path, buckets_limit=20, data_limit=None, num_signals_per_file=1000)