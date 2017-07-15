# coding: utf-8
from __future__ import division
from __future__ import print_function
import os, codecs, re, sys, math
import chainer
import numpy as np
import scipy.io.wavfile as wavfile
import pickle
import config
import acoustics
import fft
from chainer import cuda
from util import stdout, print_bold

wav_path_list = [
	"/home/stark/aibo/CSJ/WAV/core",
	"/home/stark/aibo/CSJ/WAV/noncore",
]
transcription_path_list = [
	"/home/stark/aibo/CSJ_/core",
	"/home/stark/aibo/CSJ_/noncore",
]
cache_path = "/home/stark/aibo/wav"


def get_vocab():
	characters = [
		u"_",	# blank
		u"あ",u"い",u"う",u"え",u"お",
		u"か",u"き",u"く",u"け",u"こ",
		u"さ",u"し",u"す",u"せ",u"そ",
		u"た",u"ち",u"つ",u"て",u"と",
		u"な",u"に",u"ぬ",u"ね",u"の",
		u"は",u"ひ",u"ふ",u"へ",u"ほ",
		u"ま",u"み",u"む",u"め",u"も",
		u"や",u"ゆ",u"よ",
		u"ら",u"り",u"る",u"れ",u"ろ",
		u"わ",u"を",u"ん",
		u"が",u"ぎ",u"ぐ",u"げ",u"ご",
		u"ざ",u"じ",u"ず",u"ぜ",u"ぞ",
		u"だ",u"ぢ",u"づ",u"で",u"ど",
		u"ば",u"び",u"ぶ",u"べ",u"ぼ",
		u"ぱ",u"ぴ",u"ぷ",u"ぺ",u"ぽ",
		u"ぁ",u"ぃ",u"ぅ",u"ぇ",u"ぉ",
		u"ゃ",u"ゅ",u"ょ",
		u"っ",
		u"ー",
	]

	vocab = {}
	for char in characters:
		vocab[char] = len(vocab)

	vocab_inv = {}
	for char, char_id in vocab.items():
		vocab_inv[char_id] = char

	id_blank = 0

	return vocab, vocab_inv, id_blank

def get_bucket_idx(signal, sampling_rate=16000, split_sec=0.5):
	divider = sampling_rate * split_sec
	return int(len(signal) // divider)

def extract_features(signal, sampling_rate=16000, num_fft=512, frame_width=0.032, frame_shift=0.01, num_mel_filters=40, window_func=lambda x:np.hanning(x), using_delta=True, using_delta_delta=True):
	# メルフィルタバンク出力の対数を計算
	logmel, energy = fbank(signal, sampling_rate, nfft=num_fft, winlen=frame_width, winstep=frame_shift, nfilt=num_mel_filters, winfunc=window_func)
	logmel = np.log(logmel)

	# ΔとΔΔを計算
	delta = (np.roll(logmel, -1, axis=0) - logmel) / 2 if using_delta else None
	delta_delta = (np.roll(delta, -1, axis=0) - delta) / 2 if using_delta_delta else None

	# 不要な部分を削除
	# ΔΔまで計算すると末尾の2つは正しくない値になる
	logmel = logmel[:-2].T
	delta = delta[:-2].T if using_delta else None
	delta_delta = delta_delta[:-2].T if using_delta_delta else None

	return logmel, delta, delta_delta

def generate_buckets(wav_paths, transcription_paths, cache_path, buckets_limit, data_limit, num_signals_per_file=1000):
	assert len(wav_paths) > 0
	assert len(transcription_paths) > 0

	config = chainer.config
	vocab = get_vocab()[0]
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

	def compute_mean_and_std(bucket_idx):
		num_signals = len(buckets_signal[bucket_idx])
		assert num_signals > 0
		mean = 0
		std = 0

		for signal in buckets_signal[bucket_idx]:

			specgram = fft.get_specgram(signal, config.sampling_rate, nfft=config.num_fft, winlen=config.frame_width, winstep=config.frame_shift, winfunc=config.window_func)
			logmel = fft.compute_logmel(specgram, config.sampling_rate, nfft=config.num_fft, winlen=config.frame_width, winstep=config.frame_shift, nfilt=config.num_mel_filters, winfunc=config.window_func)
			logmel, delta, delta_delta = fft.compute_deltas(logmel)
			logmel = logmel.T
			delta = delta.T
			delta_delta = delta_delta.T

			if logmel.shape[1] == 0:
				continue
			div = 1
			if config.using_delta:
				logmel = np.concatenate((logmel, delta), axis=1)
				div += 1
			if config.using_delta_delta:
				logmel = np.concatenate((logmel, delta_delta), axis=1)
				div += 1
			logmel = np.reshape(logmel, (config.num_mel_filters, div, -1))
			mean += np.mean(logmel, axis=2, keepdims=True) / num_signals
			std += np.std(logmel, axis=2, keepdims=True) / num_signals

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

			# 転記の読み込み
			batch = []
			with codecs.open(os.path.join(trn_dir, trn_filename), "r", "utf-8") as f:
				for data in f:
					period_str, channel, sentence_str = data.split(":")
					period = period_str.split("-")
					start_sec, end_sec = float(period[0]), float(period[1])
					start_frame = int(start_sec * sampling_rate)
					end_frame = int(end_sec * sampling_rate)

					assert start_frame <= len(audio)
					assert end_frame <= len(audio)

					signal = audio[start_frame:end_frame]

					assert len(signal) == end_frame - start_frame

					# channelに従って選択
					if signal.ndim == 2:
						if channel == "S":	# 両方に含まれる場合
							signal = signal[:, 0]
						elif channel == "L":
							signal = signal[:, 0]
						elif channel == "R":
							signal = signal[:, 1]
						else:
							raise Exception()

					# 文字IDに変換
					sentence = []
					sentence_str = sentence_str.strip()
					for char in sentence_str:
						if char not in vocab:
							continue
						char_id = vocab[char]
						sentence.append(char_id)

					batch.append((signal, sentence))

			# 信号長と転記文字列長の不自然な部分を検出
			num_points_per_character = 0	# 1文字あたりの信号の数
			for signal, sentence in batch:
				num_points_per_character += len(signal) / len(sentence)
			num_points_per_character /= len(signal)

			accept_rate = 0.4	# ズレの割合がこれ以下なら教師データに誤りが含まれている可能性があるので目視で確認すべき
			if trn_filename == "M03F0017.trn":	# CSJのこのファイルだけ異常な早口がある
				accept_rate = 0.05
			for idx, (signal, sentence) in enumerate(batch):
				error = abs(len(signal) - num_points_per_character * len(sentence))
				rate = error / len(signal)
				if rate < accept_rate:
					raise Exception(len(signal), len(sentence), num_points_per_character, rate, trn_filename, idx + 1)
				
				write_to_file, bucket_idx = add_to_bukcet(signal, sentence)
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

class AugmentationOption():
	def __init__(self):
		self.change_vocal_tract = False
		self.change_speech_rate = False
		self.add_noise = False

	def using_augmentation(self):
		if self.change_vocal_tract:
			return True
		if self.change_speech_rate:
			return True
		if self.add_noise:
			return True
		return False

class DevMinibatchIterator(object):
	def __init__(self, dataset, batchsizes, option=None, gpu=True):
		self.dataset = dataset
		self.batchsizes = batchsizes
		self.option = None
		self.gpu = gpu
		self.bucket_idx = 0
		self.group_idx = 0
		self.pos = 0

	def __iter__(self):
		return self

	def __next__(self):
		bucket_idx = self.bucket_idx
		group_idx = self.group_idx

		if bucket_idx >= len(self.dataset.buckets_indices_dev):
			raise StopIteration()

		signal_list = self.dataset.get_signals_for_bucket_and_group(bucket_idx, group_idx)
		sentence_list = self.dataset.get_sentences_for_bucket_and_group(bucket_idx, group_idx)
				
		indices_dev = self.dataset.buckets_indices_dev[bucket_idx][group_idx]

		batchsize = self.batchsizes[bucket_idx]
		batchsize = len(indices_dev) - self.pos if batchsize > len(indices_dev) - self.pos else batchsize
		indices = indices_dev[self.pos:self.pos + batchsize]
		assert len(indices) > 0

		extracted_features, sentences, max_feature_length, max_sentence_length = self.dataset.extract_features_by_indices(indices, signal_list, sentence_list, option=self.option)
		x_batch, x_length_batch, t_batch, t_length_batch = self.dataset.features_to_minibatch(extracted_features, sentences, max_feature_length, max_sentence_length, gpu=self.gpu)

		self.pos += batchsize
		if self.pos >= len(indices_dev):
			self.group_idx += 1
			self.pos = 0

		if self.group_idx >= len(self.dataset.buckets_indices_dev[bucket_idx]):
			self.group_idx = 0
			self.bucket_idx += 1

		return x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx, group_idx

	next = __next__  # Python 2
		
class Dataset(object):
	def __init__(self, data_path, buckets_limit=None, num_signals_per_file=1000, num_buckets_to_store_memory=200, dev_split=0.01, seed=0, id_blank=0):
		self.num_signals_per_file = num_signals_per_file
		self.num_signals_memory = num_buckets_to_store_memory
		self.dev_split = dev_split
		self.data_path = data_path
		self.buckets_limit = buckets_limit
		self.id_blank = 0

		signal_path = os.path.join(data_path, "signal")
		sentence_path = os.path.join(data_path, "sentence")
		signal_files = os.listdir(signal_path)
		sentence_files = os.listdir(sentence_path)

		mean_filename = os.path.join(data_path, "mean.npy")
		std_filename = os.path.join(data_path, "std.npy")

		try:
			if len(signal_files) == 0:
				raise Exception()
			if len(sentence_files) == 0:
				raise Exception()
			if len(signal_files) != len(sentence_files):
				raise Exception()
			if os.path.isfile(mean_filename) == False:
				raise Exception()
			if os.path.isfile(std_filename) == False:
				raise Exception()
		except:
			raise Exception("Run dataset.py before starting training.")

		self.mean = np.load(mean_filename)[None, ...].astype(np.float32)
		self.std = np.load(std_filename)[None, ...].astype(np.float32)

		buckets_signal = []
		buckets_sentence = []
		buckets_num_data = []
		for filename in signal_files:
			pattern = r"([0-9]+)_([0-9]+)_([0-9]+)\.bucket"
			m = re.match(pattern , filename)
			if m:
				bucket_idx = int(m.group(1))
				group_idx = int(m.group(2))
				num_data = int(m.group(3))
				while len(buckets_signal) <= bucket_idx:
					buckets_signal.append([])
					buckets_sentence.append([])
					buckets_num_data.append([])
				while len(buckets_signal[bucket_idx]) <= group_idx:
					buckets_signal[bucket_idx].append(None)
					buckets_sentence[bucket_idx].append(None)
					buckets_num_data[bucket_idx].append(0)
				buckets_num_data[bucket_idx][group_idx] = num_data

		if buckets_limit is not None:
			buckets_signal = buckets_signal[:buckets_limit]
			buckets_sentence = buckets_sentence[:buckets_limit]
			buckets_num_data = buckets_num_data[:buckets_limit]

		buckets_num_group = []
		for bucket in buckets_signal:
			buckets_num_group.append(len(bucket))
		total_groups = sum(buckets_num_group)
		total_buckets = len(buckets_signal)

		np.random.seed(seed)
		buckets_indices_train = []
		buckets_indices_dev = []
		for bucket_idx in range(total_buckets):
			num_groups = buckets_num_group[bucket_idx]
			indices_train = []
			indices_dev = []
			for group_idx in range(num_groups):
				num_data = buckets_num_data[bucket_idx][group_idx]
				indices = np.arange(num_data)
				np.random.shuffle(indices)
				num_dev = int(num_data * dev_split)
				indices_train.append(indices[num_dev:])
				indices_dev.append(indices[:num_dev])
			buckets_indices_train.append(indices_train)
			buckets_indices_dev.append(indices_dev)

		self.buckets_signal = buckets_signal
		self.buckets_sentence = buckets_sentence
		self.buckets_num_group = buckets_num_group
		self.buckets_num_data = buckets_num_data
		self.buckets_indices_train = buckets_indices_train
		self.buckets_indices_dev = buckets_indices_dev

		self.total_groups = total_groups
		self.total_buckets = total_buckets
		self.bucket_distribution = np.asarray(buckets_num_group) / total_groups
		self.cached_indices = []

	def get_total_training_iterations(self, batchsizes):
		num_buckets = len(self.buckets_signal)
		batchsizes = batchsizes[:num_buckets]
		itr = 0
		for indices_group_train, batchsize in zip(self.buckets_indices_train, batchsizes):
			for indices_train in indices_group_train:
				itr += int(math.ceil(len(indices_train) / batchsize))
		return itr

	def dump_information(self):
		print_bold("bucket	#train	#dev	sec")
		total_train = 0
		total_dev = 0
		config = chainer.config
		for bucket_idx, (indices_group_train, indices_group_dev) in enumerate(zip(self.buckets_indices_train, self.buckets_indices_dev)):
			if self.buckets_limit is not None and bucket_idx >= self.buckets_limit:
				break
			num_train = 0
			num_dev = 0
			for indices_train in indices_group_train:
				total_train += len(indices_train)
				num_train += len(indices_train)
			for indices_dev in indices_group_dev:
				total_dev += len(indices_dev)
				num_dev += len(indices_dev)
			print("{}	{:>6}	{:>4}	{:>6.3f}".format(bucket_idx + 1, num_train, num_dev, config.bucket_split_sec * (bucket_idx + 1)))
		print("total	{:>6}	{:>4}".format(total_train, total_dev))

	def get_signals_for_bucket_and_group(self, bucket_idx, group_idx):
		num_data = self.buckets_num_data[bucket_idx][group_idx]
		signal_list = self.buckets_signal[bucket_idx][group_idx]
		if signal_list is None:
			with open(os.path.join(self.data_path, "signal", "{}_{}_{}.bucket".format(bucket_idx, group_idx, num_data)), "rb") as f:
				signal_list = pickle.load(f)
				self.buckets_signal[bucket_idx][group_idx] = signal_list

		# 一定以上はメモリ解放
		self.cached_indices.append((bucket_idx, group_idx))
		if len(self.cached_indices) > self.num_signals_memory:
			_bucket_idx, _group_idx = self.cached_indices.pop(0)
			self.buckets_signal[_bucket_idx][_group_idx] = None
			self.buckets_sentence[bucket_idx][group_idx] = None

		return signal_list

	def get_sentences_for_bucket_and_group(self, bucket_idx, group_idx):
		num_data = self.buckets_num_data[bucket_idx][group_idx]
		sentence_list = self.buckets_sentence[bucket_idx][group_idx]
		if sentence_list is None:
			with open (os.path.join(self.data_path, "sentence", "{}_{}_{}.bucket".format(bucket_idx, group_idx, num_data)), "rb") as f:
				sentence_list = pickle.load(f)
				self.buckets_sentence[bucket_idx][group_idx] = sentence_list
		return sentence_list

	def features_to_minibatch(self, features, sentences, max_feature_length, max_sentence_length, gpu=True):
		config = chainer.config
		batchsize = len(features)
		channels = 1
		if config.using_delta:
			channels += 1
		if config.using_delta_delta:
			channels += 1
		height = config.num_mel_filters

		x_batch = np.zeros((batchsize, channels, height, max_feature_length), dtype=np.float32)
		t_batch = np.full((batchsize, max_sentence_length), self.id_blank, dtype=np.int32)
		x_length_batch = []
		t_length_batch = []

		for batch_idx, ((logmel, delta, delta_delta), sentence) in enumerate(zip(features, sentences)):
			x_length = logmel.shape[1]
			t_length = len(sentence)

			x_batch[batch_idx, 0, :, :x_length] = logmel
			if config.using_delta:
				x_batch[batch_idx, 1, :, :x_length] = delta
			if config.using_delta_delta:
				x_batch[batch_idx, 2, :, :x_length] = delta_delta
			x_length_batch.append(x_length)

			# CTCが適用可能かチェック
			num_trans_same_label = np.count_nonzero(sentence == np.roll(sentence, 1))
			required_length = t_length * 2 + 1 + num_trans_same_label
			if x_length < required_length:
				possibole_t_length = (x_length - num_trans_same_label - 1) // 2
				sentence = sentence[:possibole_t_length]
				t_length = len(sentence)

			# t
			t_batch[batch_idx, :t_length] = sentence
			t_length_batch.append(t_length)

		x_batch = (x_batch - self.mean) / self.std

		# GPU
		if gpu:
			x_batch = cuda.to_gpu(x_batch.astype(np.float32))
			t_batch = cuda.to_gpu(t_batch.astype(np.int32))
			x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
			t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

		return x_batch, x_length_batch, t_batch, t_length_batch

	def extract_features_by_indices(self, indices, signal_list, sentence_list, option=None):
		config = chainer.config
		max_feature_length = 0
		max_sentence_length = 0
		extracted_features = []
		sentences = []

		for data_idx in indices:
			signal = signal_list[data_idx]
			sentence = sentence_list[data_idx]

			# データ拡大
			if option is not None and option.using_augmentation():
				if option.add_noise:
					gain = max(min(np.random.normal(200, 100), 500), 0)
					noise = acoustics.generator.noise(len(signal), color="white") * gain
					signal += noise.astype(np.int16)

			specgram = fft.get_specgram(signal, config.sampling_rate, nfft=config.num_fft, winlen=config.frame_width, winstep=config.frame_shift, winfunc=config.window_func)
			
			# データ拡大
			if option is not None and option.using_augmentation():
				specgram = fft.augment_specgram(specgram, option.change_speech_rate, option.change_vocal_tract)

			logmel = fft.compute_logmel(specgram, config.sampling_rate, nfft=config.num_fft, winlen=config.frame_width, winstep=config.frame_shift, nfilt=config.num_mel_filters, winfunc=config.window_func)
			logmel, delta, delta_delta = fft.compute_deltas(logmel)
			logmel = logmel.T
			delta = delta.T
			delta_delta = delta_delta.T

			if logmel.shape[1] > max_feature_length:
				max_feature_length = logmel.shape[1]
			if len(sentence) > max_sentence_length:
				max_sentence_length = len(sentence)

			if logmel.shape[1] == 0:
				continue

			extracted_features.append((logmel, delta, delta_delta))
			sentences.append(sentence)

		assert max_feature_length > 0
		return extracted_features, sentences, max_feature_length, max_sentence_length

	def get_minibatch(self, batchsizes, option=None, gpu=True):
		bucket_idx = np.random.choice(np.arange(len(self.buckets_signal)), size=1, p=self.bucket_distribution)[0]
		group_idx = np.random.choice(np.arange(self.buckets_num_group[bucket_idx]), size=1)[0]

		signal_list = self.get_signals_for_bucket_and_group(bucket_idx, group_idx)
		sentence_list = self.get_sentences_for_bucket_and_group(bucket_idx, group_idx)
	
		indices = self.buckets_indices_train[bucket_idx][group_idx]
		np.random.shuffle(indices)

		batchsize = batchsizes[bucket_idx]
		batchsize = len(indices) if batchsize > len(indices) else batchsize
		indices = indices[:batchsize]

		extracted_features, sentences, max_feature_length, max_sentence_length = self.extract_features_by_indices(indices, signal_list, sentence_list, option=option)
		x_batch, x_length_batch, t_batch, t_length_batch = self.features_to_minibatch(extracted_features, sentences, max_feature_length, max_sentence_length, gpu=gpu)

		return x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx

def mkdir(d):
	try:
		os.mkdir(d)
	except:
		pass

if __name__ == "__main__":
	mkdir(cache_path)
	mkdir(os.path.join(cache_path, "signal"))
	mkdir(os.path.join(cache_path, "sentence"))

	# すべての.wavを読み込み、一定の長さごとに保存
	generate_buckets(wav_path_list, transcription_path_list, cache_path, 20, None, 1000)