import os, pickle
import numpy as np
from chainer import cuda
from ..readers.buckets import Reader
from ..processing import Processor
from ...utils import stdout, printb, Object
from .. import iterators

class Loader():
	def __init__(self):
		self.stats_total = 0
		self.stats_mean = None	# データの平均の近似
		self.stats_nvar = None	# データの分散（×データ数）の近似
		self.apply_cmn = False

	def features_to_minibatch(self, features, sentences, max_feature_length, max_sentence_length, gpu=True):
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.processor.features_to_minibatch(features, sentences, max_feature_length, max_sentence_length, self.token_ids, 
			self.id_blank)

		if self.stats_total > 0:
			for x, length in zip(x_batch, x_length_batch):
				self._update_stats_recursively(x[..., :length])
			x_mean, x_std = self.get_mean_and_std()
			x_batch = (x_batch - x_mean) / x_std

		if gpu:
			x_batch = cuda.to_gpu(x_batch.astype(np.float32))
			t_batch = cuda.to_gpu(t_batch.astype(np.int32))
			bigram_batch = cuda.to_gpu(bigram_batch.astype(np.int32))
			x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
			t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch

	def extract_batch_features(self, batch, augmentation=None):
		return self.processor.extract_batch_features(batch, augmentation, self.apply_cmn)

	# 標本平均と不偏標準偏差を返す
	def get_mean_and_std(self):
		xp = cuda.get_array_module(self.stats_mean)
		return self.stats_mean[None, ..., None], xp.sqrt(self.stats_nvar[None, ..., None] / (self.stats_total - 1))

	def update_stats(self, iteration, batchsizes, augmentation=None):
		# stack = None
		for i in range(iteration):
			batch, bucket_idx, piece_id = self.reader.sample_minibatch(batchsizes)
			audio_features, sentences, max_feature_length, max_sentence_length = self.extract_batch_features(batch, augmentation=augmentation)
			x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.processor.features_to_minibatch(audio_features, sentences, max_feature_length, max_sentence_length, self.token_ids, self.id_blank)

			# xp = cuda.get_array_module(x_batch)
			for x, length in zip(x_batch, x_length_batch):
				# if stack is None:
				# 	stack = x[..., :length]
				# else:
				# 	stack = xp.concatenate((stack, x[..., :length]), axis=2)
				self._update_stats_recursively(x[..., :length])

		# x_mean, x_std = self.get_mean_and_std()
		# true_mean = np.mean(stack, axis=2)
		# true_std = np.std(stack, axis=2)
		# print(xp.mean(abs(true_mean - x_mean), axis=(0, 2)))
		# print(xp.mean(abs(true_std - x_std), axis=(0, 2)))

	def _update_stats_recursively(self, x_batch):
		batchsize = x_batch.shape[2]
		if self.stats_total == 0:
			self.stats_mean = np.mean(x_batch, axis=2)
			self.stats_nvar = np.var(x_batch, axis=2) * batchsize
		else:
			old_mean = self.stats_mean
			old_nvar = self.stats_nvar
			sample_sum = np.sum(x_batch, axis=2)
			sample_squared_sum = np.sum(x_batch ** 2, axis=2)

			new_mean = old_mean + (sample_sum - batchsize * old_mean) / (self.stats_total + batchsize)
			new_nvar = old_nvar + sample_squared_sum - sample_sum * (new_mean + old_mean) + batchsize * new_mean * old_mean

			self.stats_mean = new_mean
			self.stats_nvar = new_nvar
		self.stats_total += batchsize

	def save_stats(self, directory):
		try:
			os.mkdir(directory)
		except:
			pass
		np.save(os.path.join(directory, "mean.npy"), self.stats_mean)
		np.save(os.path.join(directory, "nvar.npy"), self.stats_nvar)
		with open(os.path.join(directory, "total.count"), mode="wb") as f:
			pickle.dump(self.stats_total, f)

	def load_stats(self, directory):
		mean_filename = os.path.join(directory, "mean.npy")
		nvar_filename = os.path.join(directory, "nvar.npy")
		total_filename = os.path.join(directory, "total.count")

		if os.path.isfile(mean_filename) is False:
			return False
		if os.path.isfile(nvar_filename) is False:
			return False
		if os.path.isfile(total_filename) is False:
			return False

		self.stats_mean = np.load(mean_filename).astype(np.float32)
		self.stats_nvar = np.load(nvar_filename).astype(np.float32)
		with open(total_filename, mode="rb") as f:
			self.stats_total = pickle.load(f)

		return True
