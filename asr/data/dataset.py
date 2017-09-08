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

	def set_batchsizes_train(self, batchsizes):
		self.batchsizes_train = batchsizes

	def set_batchsizes_dev(self, batchsizes):
		self.batchsizes_dev = batchsizes

	def get_training_batch_iterator(self, batchsizes, augmentation=None, gpu=True):
		return TrainingBatchIterator(self, batchsizes, augmentation, gpu)

	def get_development_batch_iterator(self, batchsizes, augmentation=None, gpu=True):
		return DevelopmentBatchIterator(self, batchsizes, augmentation, gpu)

	def features_to_minibatch(self, processor, features, sentences, max_feature_length, max_sentence_length, gpu=True):
		return processor.features_to_minibatch(features, sentences, max_feature_length, max_sentence_length, self.token_ids, self.id_blank, self.mean, self.std, gpu)

	def extract_batch_features(self, processor, batch, augmentation=None):
		return processor.extract_batch_features(batch, augmentation, self.apply_cmn)

	def sample_minibatch(self, processor, augmentation=None, gpu=True):
		batch, bucket_idx, group_idx = self.reader.sample_minibatch(self.batchsizes_train)
		audio_features, sentences, max_feature_length, max_sentence_length = self.extract_batch_features(processor, batch, augmentation=augmentation)
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.features_to_minibatch(processor, audio_features, sentences, max_feature_length, max_sentence_length, gpu=gpu)

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx, group_idx

	def get_statistics(self):
		content = ""
		content += self.reader.get_statistics()
		return content

	def dump(self):
		printb("[Dataset]")
		self.reader.dump()
