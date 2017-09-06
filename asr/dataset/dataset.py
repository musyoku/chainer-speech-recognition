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
from . import preprocessing
from .reader import BucketsReader
from .. import fft
from ..utils import stdout, printb
from ..vocab import convert_sentence_to_unigram_tokens

def get_bucket_index(signal, sampling_rate=16000, split_sec=0.5):
	divider = sampling_rate * split_sec
	return int(len(signal) // divider)

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
			raise Exception("Run preprocess/buckets.py before starting training.")

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

		extracted_features, sentences, max_feature_length, max_sentence_length = extract_features_by_indices(indices, self.buckets_signal[bucket_idx], self.buckets_sentence[bucket_idx], option=self.option, apply_cmn=self.apply_cmn)
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = features_to_minibatch(extracted_features, sentences, max_feature_length, max_sentence_length, self.token_ids, self.id_blank, self.mean, self.std, gpu=self.gpu)

		self.pos += batchsize
		if self.pos >= num_data:
			self.pos = 0
			self.bucket_idx += 1

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx, self.pos / num_data

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
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.dataset.features_to_minibatch(extracted_features, sentences, max_feature_length, max_sentence_length, gpu=self.gpu)

		self.pos += batchsize
		if self.pos >= len(indices_dev):
			self.group_idx += 1
			self.pos = 0

		if self.group_idx >= len(buckets_indices[bucket_idx]):
			self.group_idx = 0
			self.bucket_idx += 1

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx, group_idx

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
	def __init__(self, data_path, batchsizes, buckets_limit=None, num_signals_per_file=1000, num_buckets_to_store_memory=200, 
		token_ids=None, dev_split=0.01, seed=0, id_blank=0, apply_cmn=False, global_normalization=True):
		assert token_ids is not None
		self.num_signals_per_file = num_signals_per_file
		self.num_signals_memory = num_buckets_to_store_memory
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
