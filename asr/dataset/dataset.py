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
from ..utils import stdout, printb, Object
from ..vocab import convert_sentence_to_unigram_tokens

class AugmentationOption(Object):
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
	def __init__(self, data_path, batchsizes_train, batchsizes_dev, buckets_limit=None, buckets_cache_size=200, token_ids=None, dev_split=0.01, seed=0, 
		id_blank=0, apply_cmn=False, global_normalization=True):
		assert token_ids is not None
		assert isinstance(token_ids, dict)

		self.data_path = data_path
		self.id_blank = 0
		self.token_ids = token_ids
		self.batchsizes_train = batchsizes_train
		self.batchsizes_dev = batchsizes_dev
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

	def set_batchsizes_train(self, batchsizes):
		self.batchsizes_train = batchsizes

	def set_batchsizes_dev(self, batchsizes):
		self.batchsizes_dev = batchsizes

	def get_training_batch_iterator(self, batchsizes, augmentation=None, gpu=True):
		return TrainingBatchIterator(self, batchsizes, augmentation, gpu)

	def get_development_batch_iterator(self, batchsizes, augmentation=None, gpu=True):
		return DevelopmentBatchIterator(self, batchsizes, augmentation, gpu)

	def get_total_training_iterations(self):
		return self.reader.calculate_total_training_iterations_with_batchsizes(self.batchsizes_train)

	def get_total_dev_iterations(self):
		return self.reader.calculate_total_dev_iterations_with_batchsizes(self.batchsizes_dev)

	def get_num_buckets(self):
		return self.reader.get_num_buckets()

	def features_to_minibatch(self, features, sentences, max_feature_length, max_sentence_length, gpu=True):
		return processing.features_to_minibatch(features, sentences, max_feature_length, max_sentence_length, self.token_ids, self.id_blank, self.mean, self.std, gpu)

	def extract_batch_features(self, batch, augmentation=None):
		return processing.extract_batch_features(batch, augmentation, self.apply_cmn, self.fbank)

	def sample_minibatch(self, augmentation=None, gpu=True):
		batch, bucket_idx, group_idx = self.reader.sample_minibatch(self.batchsizes_train)
		audio_features, sentences, max_feature_length, max_sentence_length = self.extract_batch_features(batch, augmentation=augmentation)
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.features_to_minibatch(audio_features, sentences, max_feature_length, max_sentence_length, gpu=gpu)

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx, group_idx

	def get_statistics(self):
		content = ""
		content += self.reader.get_statistics()
		return content

	def dump(self):
		printb("[Dataset]")
		self.reader.dump()
