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
from util import stdout, printb
from vocab import convert_sentence_to_unigram_ids

wav_path_list = [
	"/home/stark/sandbox/CSJ/WAV/core",
	"/home/stark/sandbox/CSJ/WAV/noncore",
]
transcription_path_list = [
	"/home/stark/sandbox/CSJ_/core",
	"/home/stark/sandbox/CSJ_/noncore",
]
cache_path = "/home/stark/sandbox/wav"

wav_path_test = "/home/stark/sandbox/CSJ/WAV/test"
trn_path_test = "/home/stark/sandbox/CSJ_/test"

def get_bucket_index(signal, sampling_rate=16000, split_sec=0.5):
	divider = sampling_rate * split_sec
	return int(len(signal) // divider)

def load_test_buckets(wav_dir, trn_dir, buckets_limit=None):
	config = chainer.config
	buckets_signal = []
	buckets_sentence = []
	vocab = get_vocab()[0]
	total_min = 0

	def append_bucket(buckets, bucket_idx, data):
		if len(buckets) <= bucket_idx:
			while len(buckets) <= bucket_idx:
				buckets.append([])
		buckets[bucket_idx].append(data)

	def add_to_bukcet(signal, sentence):
		bucket_idx = get_bucket_index(signal, config.sampling_rate, config.bucket_split_sec)
		if buckets_limit and bucket_idx >= buckets_limit:
			return bucket_idx
		append_bucket(buckets_signal, bucket_idx, signal)
		append_bucket(buckets_sentence, bucket_idx, sentence)
		return bucket_idx

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

	for data_idx, wav_id in enumerate(wav_ids):
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
		sys.stdout.write("Loading {} ({}/{}) ... shape={}, rate={}, min={}".format(wav_filename, data_idx + 1, len(wav_fs), audio.shape, sampling_rate, int(duration)))
		sys.stdout.flush()

		# 転記の読み込みと音声の切り出し
		batch = generate_signal_transcription_pairs(os.path.join(trn_dir, trn_filename), audio, sampling_rate, vocab)

		# 信号長と転記文字列長の不自然な部分を検出
		num_points_per_character = 0	# 1文字あたりの信号の数
		for signal, sentence in batch:
			num_points_per_character += len(signal) / len(sentence)
		num_points_per_character /= len(signal)

		accept_rate = 0.7	# ズレの割合がこれ以下なら教師データに誤りが含まれている可能性があるので目視で確認すべき

		for idx, (signal, sentence) in enumerate(batch):
			error = abs(len(signal) - num_points_per_character * len(sentence))
			rate = error / len(signal)
			if rate < accept_rate:
				raise Exception(len(signal), len(sentence), num_points_per_character, rate, trn_filename, idx + 1)
			
			add_to_bukcet(signal, sentence)

	sys.stdout.write("\r")
	sys.stdout.write(stdout.CLEAR)
	printb("bucket	#data	sec")
	total = 0
	for bucket_idx, signals in enumerate(buckets_signal):
		if buckets_limit and bucket_idx >= buckets_limit:
			break
		num_data = len(signals)
		total += num_data
		print("{}	{:>4}	{:>6.3f}".format(bucket_idx + 1, num_data, config.bucket_split_sec * (bucket_idx + 1)))
	print("total	{:>4}		{} hour".format(total, int(total_min / 60)))

	return buckets_signal, buckets_sentence

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

			unigram_ids = convert_sentence_to_unigram_ids(sentence)

			batch.append((signal, unigram_ids, sentence))
	return batch

def extract_features_by_indices(indices, signal_list, sentence_list, option=None, fbank=None):
	config = chainer.config
	max_feature_length = 0
	max_sentence_length = 0
	extracted_features = []
	sentences = []

	for data_idx in indices:
		signal = signal_list[data_idx]
		sentence = sentence_list[data_idx]

		# データ拡大
		if option and option.add_noise:
			gain = max(min(np.random.normal(200, 100), 500), 0)
			noise = acoustics.generator.noise(len(signal), color="white") * gain
			signal += noise.astype(np.int16)

		specgram = fft.get_specgram(signal, config.sampling_rate, nfft=config.num_fft, winlen=config.frame_width, winstep=config.frame_shift, winfunc=config.window_func)
		
		# データ拡大
		if option and option.using_augmentation():
			specgram = fft.augment_specgram(specgram, option.change_speech_rate, option.change_vocal_tract)

		logmel = fft.compute_logmel(specgram, config.sampling_rate, fbank=fbank, nfft=config.num_fft, winlen=config.frame_width, winstep=config.frame_shift, nfilt=config.num_mel_filters, winfunc=config.window_func)
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

def features_to_minibatch(features, sentences, max_feature_length, max_sentence_length, id_blank, x_mean, x_std, gpu=True):
	config = chainer.config
	batchsize = len(features)
	channels = 1
	if config.using_delta:
		channels += 1
	if config.using_delta_delta:
		channels += 1
	height = config.num_mel_filters

	x_batch = np.zeros((batchsize, channels, height, max_feature_length), dtype=np.float32)
	t_batch = np.full((batchsize, max_sentence_length), id_blank, dtype=np.int32)
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

	x_batch = (x_batch - x_mean) / x_std

	# GPU
	if gpu:
		x_batch = cuda.to_gpu(x_batch.astype(np.float32))
		t_batch = cuda.to_gpu(t_batch.astype(np.int32))
		x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
		t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

	return x_batch, x_length_batch, t_batch, t_length_batch

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

class TestMinibatchIterator(object):
	def __init__(self, wav_dir, trn_dir, cache_dir, batchsizes, id_blank, buckets_limit=None, option=None, gpu=True):
		self.option = None
		self.batchsizes = batchsizes
		self.gpu = gpu
		self.bucket_idx = 0
		self.pos = 0
		self.id_blank = id_blank
		self.buckets_signal, self.buckets_sentence = load_test_buckets(wav_dir, trn_dir, buckets_limit)

		assert len(self.buckets_signal) > 0
		assert len(self.buckets_sentence) > 0

		mean_filename = os.path.join(cache_dir, "mean.npy")
		std_filename = os.path.join(cache_dir, "std.npy")

		try:
			if os.path.isfile(mean_filename) == False:
				raise Exception()
			if os.path.isfile(std_filename) == False:
				raise Exception()
		except:
			raise Exception("Run dataset.py before starting training.")

		self.mean = np.load(mean_filename)[None, ...].astype(np.float32)
		self.std = np.load(std_filename)[None, ...].astype(np.float32)

	def reset(self):
		self.bucket_idx = 0
		self.pos = 0

	def __iter__(self):
		return self

	def __next__(self):
		bucket_idx = self.bucket_idx

		if bucket_idx >= len(self.buckets_signal):
			raise StopIteration()

		num_data = len(self.buckets_signal[bucket_idx])
		while num_data == 0:
			bucket_idx += 1
			if bucket_idx >= len(self.buckets_signal):
				raise StopIteration()
			num_data = len(self.buckets_signal[bucket_idx])

		batchsize = self.batchsizes[bucket_idx]
		batchsize = num_data - self.pos if batchsize > num_data - self.pos else batchsize
		indices = np.arange(self.pos, self.pos + batchsize)
		assert len(indices) > 0

		extracted_features, sentences, max_feature_length, max_sentence_length = extract_features_by_indices(indices, self.buckets_signal[bucket_idx], self.buckets_sentence[bucket_idx], option=self.option)
		x_batch, x_length_batch, t_batch, t_length_batch = features_to_minibatch(extracted_features, sentences, max_feature_length, max_sentence_length, self.id_blank, self.mean, self.std, gpu=self.gpu)

		self.pos += batchsize
		if self.pos >= num_data:
			self.pos = 0
			self.bucket_idx += 1

		return x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx, self.pos / num_data

	next = __next__  # Python 2


class DevelopmentMinibatchIterator(object):
	def __init__(self, dataset, batchsizes, option=None, id_blank=0, gpu=True):
		self.dataset = dataset
		self.batchsizes = batchsizes
		self.option = None
		self.gpu = gpu
		self.bucket_idx = 0
		self.group_idx = 0
		self.pos = 0
		self.id_blank = 0

	def __iter__(self):
		return self

	def __next__(self):
		bucket_idx = self.bucket_idx
		group_idx = self.group_idx
		buckets_indices = self.dataset.buckets_indices_dev

		if bucket_idx >= len(buckets_indices):
			raise StopIteration()

		signal_list = self.dataset.get_signals_by_bucket_and_group(bucket_idx, group_idx)
		sentence_list = self.dataset.get_sentences_by_bucket_and_group(bucket_idx, group_idx)
				
		indices_dev = buckets_indices[bucket_idx][group_idx]

		batchsize = self.batchsizes[bucket_idx]
		batchsize = len(indices_dev) - self.pos if batchsize > len(indices_dev) - self.pos else batchsize
		indices = indices_dev[self.pos:self.pos + batchsize]
		if len(indices) == 0:
			import pdb; pdb.set_trace()
		assert len(indices) > 0

		extracted_features, sentences, max_feature_length, max_sentence_length = self.dataset.extract_features_by_indices(indices, signal_list, sentence_list, option=self.option)
		x_batch, x_length_batch, t_batch, t_length_batch = self.dataset.features_to_minibatch(extracted_features, sentences, max_feature_length, max_sentence_length, gpu=self.gpu)

		self.pos += batchsize
		if self.pos >= len(indices_dev):
			self.group_idx += 1
			self.pos = 0

		if self.group_idx >= len(buckets_indices[bucket_idx]):
			self.group_idx = 0
			self.bucket_idx += 1

		return x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx, group_idx

	next = __next__  # Python 2
		
class TrainingMinibatchIterator(object):
	def __init__(self, dataset, batchsizes, option=None, id_blank=0, gpu=True):
		self.dataset = dataset
		self.batchsizes = batchsizes
		self.option = None
		self.gpu = gpu
		self.id_blank = 0
		self.loop_count = 0
		self.total_loop = dataset.get_total_training_iterations()

	def __iter__(self):
		return self

	def __next__(self):
		if self.loop_count >= self.total_loop:
			raise StopIteration()
		self.loop_count += 1
		return dataset.get_minibatch(self.option, self.gpu)

	next = __next__  # Python 2

class Dataset(object):
	def __init__(self, data_path, batchsizes, buckets_limit=None, num_signals_per_file=1000, num_buckets_to_store_memory=200, dev_split=0.01, seed=0, id_blank=0):
		self.num_signals_per_file = num_signals_per_file
		self.num_signals_memory = num_buckets_to_store_memory
		self.dev_split = dev_split
		self.data_path = data_path
		self.buckets_limit = buckets_limit
		self.id_blank = 0
		self.batchsizes = batchsizes

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
		buckets_num_updates = []
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
					buckets_num_updates.append([])
				while len(buckets_signal[bucket_idx]) <= group_idx:
					buckets_signal[bucket_idx].append(None)
					buckets_sentence[bucket_idx].append(None)
					buckets_num_data[bucket_idx].append(0)
					buckets_num_updates[bucket_idx].append(0)
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
				if num_dev > 0:
					indices_dev.append(indices[:num_dev])
			buckets_indices_train.append(indices_train)
			buckets_indices_dev.append(indices_dev)

		self.buckets_signal = buckets_signal
		self.buckets_sentence = buckets_sentence
		self.buckets_num_group = buckets_num_group
		self.buckets_num_data = buckets_num_data
		self.buckets_num_updates = buckets_num_updates
		self.buckets_indices_train = buckets_indices_train
		self.buckets_indices_dev = buckets_indices_dev

		self.total_groups = total_groups
		self.total_buckets = total_buckets
		self.bucket_distribution = np.asarray(buckets_num_group) / total_groups
		self.cached_indices = []

		config = chainer.config
		self.fbank = fft.get_filterbanks(nfft=config.num_fft, nfilt=config.num_mel_filters, samplerate=config.sampling_rate)

	def get_iterator_train(self, batchsizes, option=None, gpu=True):
		return TrainingMinibatchIterator(self, batchsizes, option, self.id_blank, gpu)

	def get_iterator_dev(self, batchsizes, option=None, gpu=True):
		return DevelopmentMinibatchIterator(self, batchsizes, option, self.id_blank, gpu)

	def get_total_training_iterations(self):
		num_buckets = len(self.buckets_signal)
		batchsizes = self.batchsizes[:num_buckets]
		itr = 0
		for indices_group_train, batchsize in zip(self.buckets_indices_train, batchsizes):
			for indices_train in indices_group_train:
				itr += int(math.ceil(len(indices_train) / batchsize))
		return itr

	def get_signals_by_bucket_and_group(self, bucket_idx, group_idx):
		num_data = self.buckets_num_data[bucket_idx][group_idx]
		signal_list = self.buckets_signal[bucket_idx][group_idx]
		if signal_list is None:
			with open(os.path.join(self.data_path, "signal", "{}_{}_{}.bucket".format(bucket_idx, group_idx, num_data)), "rb") as f:
				signal_list = pickle.load(f)
				self.buckets_signal[bucket_idx][group_idx] = signal_list

		# 一定以上はメモリ解放
		if self.num_signals_memory > 0:
			self.cached_indices.append((bucket_idx, group_idx))
			if len(self.cached_indices) > self.num_signals_memory:
				_bucket_idx, _group_idx = self.cached_indices.pop(0)
				self.buckets_signal[_bucket_idx][_group_idx] = None
				self.buckets_sentence[bucket_idx][group_idx] = None

		return signal_list

	def get_sentences_by_bucket_and_group(self, bucket_idx, group_idx):
		num_data = self.buckets_num_data[bucket_idx][group_idx]
		sentence_list = self.buckets_sentence[bucket_idx][group_idx]
		if sentence_list is None:
			with open (os.path.join(self.data_path, "sentence", "{}_{}_{}.bucket".format(bucket_idx, group_idx, num_data)), "rb") as f:
				sentence_list = pickle.load(f)
				self.buckets_sentence[bucket_idx][group_idx] = sentence_list
		return sentence_list

	def features_to_minibatch(self, features, sentences, max_feature_length, max_sentence_length, gpu=True):
		return features_to_minibatch(features, sentences, max_feature_length, max_sentence_length, self.id_blank, self.mean, self.std, gpu)

	def extract_features_by_indices(self, indices, signal_list, sentence_list, option=None):
		return extract_features_by_indices(indices, signal_list, sentence_list, option, self.fbank)

	def get_minibatch(self, option=None, gpu=True):
		bucket_idx = np.random.choice(np.arange(len(self.buckets_signal)), size=1, p=self.bucket_distribution)[0]
		group_idx = np.random.choice(np.arange(self.buckets_num_group[bucket_idx]), size=1)[0]

		signal_list = self.get_signals_by_bucket_and_group(bucket_idx, group_idx)
		sentence_list = self.get_sentences_by_bucket_and_group(bucket_idx, group_idx)

		self.increment_num_updates(bucket_idx, group_idx)
	
		indices = self.buckets_indices_train[bucket_idx][group_idx]
		np.random.shuffle(indices)

		batchsize = self.batchsizes[bucket_idx]
		batchsize = len(indices) if batchsize > len(indices) else batchsize
		indices = indices[:batchsize]

		extracted_features, sentences, max_feature_length, max_sentence_length = self.extract_features_by_indices(indices, signal_list, sentence_list, option=option)
		x_batch, x_length_batch, t_batch, t_length_batch = self.features_to_minibatch(extracted_features, sentences, max_feature_length, max_sentence_length, gpu=gpu)

		return x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx, group_idx

	def increment_num_updates(self, bucket_idx, group_idx):
		self.buckets_num_updates[bucket_idx][group_idx] += 1

	def dump_num_updates(self):
		for bucket_idx in range(len(self.buckets_signal)):
			printb("bucket " + str(bucket_idx))
			buckets = self.buckets_num_updates[bucket_idx]
			print(buckets)
			print(sum(buckets) / len(buckets))

	def dump_information(self):
		printb("bucket	#train	#dev	sec")
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

if __name__ == "__main__":
	pass