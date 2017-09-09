import os, pickle
import numpy as np
from chainer import cuda
from ..readers.buckets import Reader
from ..processing import Processor
from ...utils import stdout, printb, Object
from .. import iterators
from . import base

class Loader(base.Loader):
	def __init__(self, data_path, batchsizes_train, batchsizes_dev=None, buckets_limit=None, bucket_split_sec=0.5,
		buckets_cache_size=200, vocab_token_to_id=None, dev_split=0.01, seed=0, id_blank=0, apply_cmn=False, 
		sampling_rate=16000, frame_width=0.032, frame_shift=0.01, num_mel_filters=40, 
		window_func="hanning", using_delta=True, using_delta_delta=True):

		assert vocab_token_to_id is not None
		assert isinstance(vocab_token_to_id, dict)

		super().__init__()

		self.batchsizes_train = batchsizes_train
		self.batchsizes_dev = batchsizes_dev
		self.token_ids = vocab_token_to_id
		self.id_blank = id_blank
		self.apply_cmn = apply_cmn

		self.processor = Processor(sampling_rate=sampling_rate, frame_width=frame_width, frame_shift=frame_shift, 
			num_mel_filters=num_mel_filters, window_func=window_func, using_delta=using_delta, using_delta_delta=using_delta_delta)

		self.reader = Reader(data_path=data_path, buckets_limit=buckets_limit, buckets_cache_size=buckets_cache_size, 
			dev_split=dev_split, seed=seed, sampling_rate=sampling_rate, bucket_split_sec=bucket_split_sec)

		self.stats_total = 0
		self.stats_mean = None	# データの平均の近似
		self.stats_nvar = None	# データの分散（×データ数）の近似

	def sample_minibatch(self, augmentation=None, gpu=True):
		# 生の音声信号を取得
		batch, bucket_idx, piece_id = self.reader.sample_minibatch(self.batchsizes_train)

		# メルフィルタバンク出力を求める
		audio_features, sentences, max_feature_length, max_sentence_length = self.extract_batch_features(batch, augmentation=augmentation)

		# 書き起こしをIDに変換
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.features_to_minibatch(audio_features, 
			sentences, max_feature_length, max_sentence_length, gpu=gpu)

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx

	def get_total_training_iterations(self):
		return self.reader.calculate_total_training_iterations_with_batchsizes(self.batchsizes_train)

	def get_total_dev_iterations(self):
		return self.reader.calculate_total_dev_iterations_with_batchsizes(self.batchsizes_dev)

	def get_num_buckets(self):
		return self.reader.get_num_buckets()

	def set_batchsizes_train(self, batchsizes):
		self.batchsizes_train = batchsizes

	def set_batchsizes_dev(self, batchsizes):
		self.batchsizes_dev = batchsizes

	def get_training_batch_iterator(self, batchsizes, augmentation=None, gpu=True):
		return iterators.buckets.train.Iterator(self, batchsizes, augmentation, gpu)

	def get_development_batch_iterator(self, batchsizes, augmentation=None, gpu=True):
		return iterators.buckets.dev.Iterator(self, batchsizes, augmentation, gpu)

	def get_statistics(self):
		content = ""
		content += self.reader.get_statistics()
		return content

	def dump(self):
		printb("[Dataset]")
		self.reader.dump()
