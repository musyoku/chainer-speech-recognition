# coding: utf-8
from __future__ import division
from __future__ import print_function
import os, codecs, re, sys, math
import chainer
import numpy as np
import scipy.io.wavfile as wavfile
import pickle
import acoustics
from chainer import cuda
from . import processing
from .reader import BucketsReader
from .iterator import TrainingBatchIterator, DevelopmentBatchIterator
from .. import fft
from ..utils import stdout, printb
from ..vocab import convert_sentence_to_unigram_tokens

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

class Dataset():
	def __init__(self, data_path, batchsizes, buckets_limit=None, buckets_cache_size=200, token_ids=None, dev_split=0.01, seed=0, 
		id_blank=0, apply_cmn=False, global_normalization=True):
		assert token_ids is not None
		assert isinstance(token_ids, dict)

		self.data_path = data_path
		self.id_blank = 0
		self.token_ids = token_ids
		self.batchsizes = batchsizes
		self.apply_cmn = apply_cmn
		self.global_normalization = global_normalization

		self.reader = BucketsReader(data_path, buckets_limit, buckets_cache_size, dev_split, seed)

		mean_filename = os.path.join(data_path, "mean.npy")
		std_filename = os.path.join(data_path, "std.npy")

		try:
			if global_normalization:
				if os.path.isfile(mean_filename) == False:
					raise Exception()
				if os.path.isfile(std_filename) == False:
					raise Exception()
		except:
			raise Exception("Run preprocess/buckets.py before starting training.")

		self.mean = np.load(mean_filename)[None, ...].astype(np.float32)
		self.std = np.load(std_filename)[None, ...].astype(np.float32)

		config = chainer.config
		self.fbank = fft.get_filterbanks(nfft=config.num_fft, nfilt=config.num_mel_filters, samplerate=config.sampling_rate)

	def get_iterator_train(self, batchsizes, option=None, gpu=True):
		return TrainingBatchIterator(self, batchsizes, option, self.id_blank, gpu)

	def get_iterator_dev(self, batchsizes, option=None, gpu=True):
		return DevelopmentBatchIterator(self, batchsizes, option, self.id_blank, gpu)

	def get_total_training_iterations(self):
		return self.reader.compute_total_training_iterations_with_batchsizes(self.batchsizes)

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
		return processing.features_to_minibatch(features, sentences, max_feature_length, max_sentence_length, self.token_ids, self.id_blank, self.mean, self.std, gpu)

	def extract_batch_features(self, batch, augmentation=None):
		return processing.extract_batch_features(batch, augmentation, self.apply_cmn, self.fbank)

	def sample_minibatch(self, augmentation=None, gpu=True):
		batch = self.reader.sample_minibatch(self.batchsizes)
		audio_features, sentences, max_feature_length, max_sentence_length = self.extract_batch_features(batch, augmentation=augmentation)
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.features_to_minibatch(audio_features, sentences, max_feature_length, max_sentence_length, gpu=gpu)

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx, group_idx

	def dump_num_updates(self):
		self.reader.dump_num_updates()

	def dump_information(self):
		self.reader.dump_information()

class _Dataset():
	def __init__(self, data_path, batchsizes, buckets_limit=None, num_signals_per_file=1000, buckets_cache_size=200, 
		token_ids=None, dev_split=0.01, seed=0, id_blank=0, apply_cmn=False, global_normalization=True):
		assert token_ids is not None
		self.num_signals_per_file = num_signals_per_file
		self.num_signals_memory = buckets_cache_size
		self.dev_split = dev_split
		self.data_path = data_path
		self.buckets_limit = buckets_limit
		self.id_blank = 0
		self.token_ids = token_ids
		self.batchsizes = batchsizes
		self.apply_cmn = apply_cmn
		self.global_normalization = global_normalization

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
			if global_normalization:
				if os.path.isfile(mean_filename) == False:
					raise Exception()
				if os.path.isfile(std_filename) == False:
					raise Exception()
		except:
			raise Exception("Run preprocess/buckets.py before starting training.")

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
		return TrainingBatchIterator(self, batchsizes, option, self.id_blank, gpu)

	def get_iterator_dev(self, batchsizes, option=None, gpu=True):
		return DevelopmentBatchIterator(self, batchsizes, option, self.id_blank, gpu)

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
		return features_to_minibatch(features, sentences, max_feature_length, max_sentence_length, self.token_ids, self.id_blank, self.mean, self.std, gpu)

	def extract_features_by_indices(self, indices, signal_list, sentence_list, option=None):
		return extract_features_by_indices(indices, signal_list, sentence_list, option, self.apply_cmn, self.fbank)

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
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.features_to_minibatch(extracted_features, sentences, max_feature_length, max_sentence_length, gpu=gpu)

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx, group_idx

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
