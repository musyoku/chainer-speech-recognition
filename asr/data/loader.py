import os
import numpy as np
from .reader import BucketsReader, AudioReader
from .processing import Processor
from .iterator import TrainingBatchIterator, DevelopmentBatchIterator
from ..utils import stdout, printb, Object

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

class AudioLoader():
	def __init__(self, wav_directory_list, transcription_directory_list, mean_filename, std_filename, batchsizes, buckets_limit=None, 
		bucket_split_sec=0.5, vocab_token_to_id=None, seed=0, id_blank=0, apply_cmn=False, sampling_rate=16000, frame_width=0.032, 
		frame_shift=0.01, num_mel_filters=40, window_func="hanning", using_delta=True, using_delta_delta=True):

		assert vocab_token_to_id is not None
		assert isinstance(vocab_token_to_id, dict)

		self.batchsizes = batchsizes
		self.token_ids = vocab_token_to_id
		self.id_blank = id_blank
		self.apply_cmn = apply_cmn

		self.processor = Processor(sampling_rate=sampling_rate, frame_width=frame_width, frame_shift=frame_shift, 
			num_mel_filters=num_mel_filters, window_func=window_func, using_delta=using_delta, using_delta_delta=using_delta_delta)

		self.reader = AudioReader(wav_directory_list=wav_directory_list, transcription_directory_list=transcription_directory_list, 
			buckets_limit=buckets_limit, frame_width=frame_width, bucket_split_sec=bucket_split_sec)

		self.mean = None
		self.std = None

		try:
			if isinstance(mean_filename, str) and isinstance(std_filename, str):
				if os.path.isfile(mean_filename) == False:
					raise Exception()
				if os.path.isfile(std_filename) == False:
					raise Exception()
				self.mean = np.load(mean_filename)[None, ...].astype(np.float32)
				self.std = np.load(std_filename)[None, ...].astype(np.float32)
		except:
			raise Exception("Run preprocess/buckets.py before starting training.")

	def dump(self):
		self.reader.dump()


class BucketsLoader():
	def __init__(self, data_path, batchsizes_train, batchsizes_dev=None, buckets_limit=None, bucket_split_sec=0.5,
		buckets_cache_size=200, vocab_token_to_id=None, dev_split=0.01, seed=0, id_blank=0, apply_cmn=False, 
		global_normalization=True, sampling_rate=16000, frame_width=0.032, frame_shift=0.01, num_mel_filters=40, 
		window_func="hanning", using_delta=True, using_delta_delta=True):

		assert vocab_token_to_id is not None
		assert isinstance(vocab_token_to_id, dict)

		self.batchsizes_train = batchsizes_train
		self.batchsizes_dev = batchsizes_dev
		self.token_ids = vocab_token_to_id
		self.id_blank = id_blank
		self.apply_cmn = apply_cmn

		self.processor = Processor(sampling_rate=sampling_rate, frame_width=frame_width, frame_shift=frame_shift, 
			num_mel_filters=num_mel_filters, window_func=window_func, using_delta=using_delta, using_delta_delta=using_delta_delta)

		self.reader = BucketsReader(data_path=data_path, buckets_limit=buckets_limit, buckets_cache_size=buckets_cache_size, 
			dev_split=dev_split, seed=seed, sampling_rate=sampling_rate, bucket_split_sec=bucket_split_sec)

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


	def features_to_minibatch(self, features, sentences, max_feature_length, max_sentence_length, gpu=True):
		return self.processor.features_to_minibatch(features, sentences, max_feature_length, max_sentence_length, self.token_ids, 
			self.id_blank, self.mean, self.std, gpu)

	def extract_batch_features(self, batch, augmentation=None):
		return self.processor.extract_batch_features(batch, augmentation, self.apply_cmn)

	def sample_minibatch(self, augmentation=None, gpu=True):
		# 生の音声信号を取得
		batch, bucket_idx, group_idx = self.reader.sample_minibatch(self.batchsizes_train)

		# メルフィルタバンク出力を求める
		audio_features, sentences, max_feature_length, max_sentence_length = self.extract_batch_features(batch, augmentation=augmentation)

		# 書き起こしをIDに変換
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.features_to_minibatch(audio_features, 
			sentences, max_feature_length, max_sentence_length, gpu=gpu)

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx, group_idx

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
		return TrainingBatchIterator(self, batchsizes, augmentation, gpu)

	def get_development_batch_iterator(self, batchsizes, augmentation=None, gpu=True):
		return DevelopmentBatchIterator(self, batchsizes, augmentation, gpu)

	def get_statistics(self):
		content = ""
		content += self.reader.get_statistics()
		return content

	def dump(self):
		printb("[Dataset]")
		self.reader.dump()
