# coding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
from dataset import load_audio_and_transcription
import chainer
import os
import numpy as np

BUCKET_THRESHOLD = 32

def get_bucket_index(length):
	return length // BUCKET_THRESHOLD

def get_minutes(length):
	config = chainer.config
	return length * config.frame_shift

def load_audio_features(buckets_limit, data_limit):
	wav_paths = [
		"/home/stark/sandbox/CSJ/WAV/core",
	]

	transcription_paths = [
		"/home/stark/sandbox/CSJ_/core",
	]

	data_cache_path = "/home/stark/sandbox/cache"

	feature_filename = os.path.join(data_cache_path, "audio.npy")
	feature_length_filename = os.path.join(data_cache_path, "audio.length.npy")
	sentence_filename = os.path.join(data_cache_path, "sentence.npy")
	sentence_length_filename = os.path.join(data_cache_path, "sentence.length.npy")
	mean_filename = os.path.join(data_cache_path, "mean.npy")
	std_filename = os.path.join(data_cache_path, "std.npy")	

	mean_x_batch = None
	stddev_x_batch = None

	if os.path.isfile(feature_filename):
		assert os.path.isfile(feature_length_filename)
		assert os.path.isfile(sentence_filename)
		assert os.path.isfile(sentence_length_filename)
		assert os.path.isfile(mean_filename)
		assert os.path.isfile(std_filename)
		print("loading {} ...".format(feature_filename))
		audio_batch = np.load(feature_filename)
		print("loading {} ...".format(feature_length_filename))
		audio_length_batch = np.load(feature_length_filename)
		print("loading {} ...".format(sentence_filename))
		sentence_batch = np.load(sentence_filename)
		print("loading {} ...".format(sentence_length_filename))
		sentence_length_batch = np.load(sentence_length_filename)
		print("loading {} ...".format(mean_filename))
		mean_x_batch = np.load(mean_filename)
		print("loading {} ...".format(std_filename))
		stddev_x_batch = np.load(std_filename)
	else:
		dataset, max_sentence_length, max_logmel_length = load_audio_and_transcription(wav_paths, transcription_paths)

		# 読み込んだデータをキャッシュ
		config = chainer.config
		try:
			os.mkdir(data_cache_path)
		except:
			pass

		# 必要なバケツの数を特定
		buckets_length = 0
		for idx, data in enumerate(dataset):
			sentence, logmel, delta, delta_delta = data
			assert logmel.shape[1] == delta.shape[1]
			assert delta.shape[1] == delta_delta.shape[1]
			audio_length = logmel.shape[1]
			bucket_index = get_bucket_index(audio_length)
			if bucket_index > buckets_length:
				buckets_length = bucket_index
		buckets_length += 1
		if buckets_limit is not None:
			buckets_length = buckets_limit if buckets_length > buckets_limit else buckets_length

		# バケツ中のデータの最大長を特定
		valid_dataset_size = 0
		max_feature_length_for_bucket = [0] * buckets_length
		max_sentence_length_for_bucket = [0] * buckets_length
		for idx, data in enumerate(dataset):
			sentence, logmel, delta, delta_delta = data
			feature_length = logmel.shape[1]
			sentence_length = len(sentence)
			bucket_index = get_bucket_index(feature_length)
			if bucket_index >= buckets_length:
				continue
			valid_dataset_size += 1
			max_feature_length_for_bucket[bucket_index] = BUCKET_THRESHOLD * (bucket_index + 1) - 1
			assert feature_length <= max_feature_length_for_bucket[bucket_index]
			if sentence_length > max_sentence_length_for_bucket[bucket_index]:
				max_sentence_length_for_bucket[bucket_index] = sentence_length

		# データの平均と標準偏差
		mean_x_batch = 0
		stddev_x_batch = 0

		# バケツにデータを格納
		buckets_feature = [None] * buckets_length
		buckets_feature_length = [None] * buckets_length
		buckets_sentence = [None] * buckets_length
		buckets_sentence_length = [None] * buckets_length
		for idx, data in enumerate(dataset):
			sentence, logmel, delta, delta_delta = data
			feature_length = logmel.shape[1]
			sentence_length = len(sentence)
			bucket_index = get_bucket_index(feature_length)
			if bucket_index >= buckets_length:
				continue

			# 音響特徴量
			feature_batch = buckets_feature[bucket_index]
			if feature_batch is None:
				max_feature_length = max_feature_length_for_bucket[bucket_index]
				feature_batch = np.zeros((3, config.num_mel_filters, max_feature_length), dtype=np.float32)
				feature_batch[0, :, :feature_length] = logmel			
				feature_batch[1, :, :feature_length] = delta			
				feature_batch[2, :, :feature_length] = delta_delta		
				# 平均と標準偏差を計算
				mean_x_batch += np.mean(feature_batch[:, :, :feature_length], axis=2, keepdims=True) / valid_dataset_size
				stddev_x_batch += np.std(feature_batch[:, :, :feature_length], axis=2, keepdims=True) / valid_dataset_size	
				# reshape
				feature_batch = feature_batch[None, ...]
			else:
				new_feature = np.zeros(feature_batch.shape[1:], dtype=np.float32)
				new_feature[0, :, :feature_length] = logmel			
				new_feature[1, :, :feature_length] = delta			
				new_feature[2, :, :feature_length] = delta_delta
				# バケツの後ろに結合
				feature_batch = np.concatenate((feature_batch, new_feature[None, ...]), axis=0)
				# 平均と標準偏差を計算
				mean_x_batch += np.mean(new_feature[:, :, :feature_length], axis=2, keepdims=True) / valid_dataset_size
				stddev_x_batch += np.std(new_feature[:, :, :feature_length], axis=2, keepdims=True) / valid_dataset_size	
			buckets_feature[bucket_index] = feature_batch

			# 書き起こし
			sentence_batch = buckets_sentence[bucket_index]
			if sentence_batch is None:
				max_sentence_length = max_sentence_length_for_bucket[bucket_index]
				sentence_batch = np.zeros((max_sentence_length,), dtype=np.int32)
				sentence_batch[:sentence_length] = sentence
				# reshape
				sentence_batch = sentence_batch[None, ...]
			else:
				new_sentence = np.zeros(sentence_batch.shape[1:], dtype=np.int32)
				new_sentence[:sentence_length] = sentence
				# バケツの後ろに結合
				sentence_batch = np.concatenate((sentence_batch, new_sentence[None, ...]), axis=0)
			buckets_sentence[bucket_index] = sentence_batch

			# 音響特徴量の有効長
			feature_length_batch = buckets_feature_length[bucket_index]
			if feature_length_batch is None:
				feature_length_batch = np.zeros((1,), dtype=np.int32)
				feature_length_batch[0] = feature_length
			else:
				feature_length_batch = np.concatenate((feature_length_batch, [feature_length]), axis=0)
			buckets_feature_length[bucket_index] = feature_length_batch

			# 書き起こしの有効長
			sentence_length_batch = buckets_sentence_length[bucket_index]
			if sentence_length_batch is None:
				sentence_length_batch = np.zeros((1,), dtype=np.int32)
				sentence_length_batch[0] = sentence_length
			else:
				sentence_length_batch = np.concatenate((sentence_length_batch, [sentence_length]), axis=0)
			buckets_sentence_length[bucket_index] = sentence_length_batch

		for bucket_index in xrange(buckets_length):
			print(buckets_feature[bucket_index].shape, buckets_sentence[bucket_index].shape, get_minutes(buckets_feature[bucket_index].shape[3]))

		# ディスクにキャッシュ
		for bucket_index in xrange(buckets_length):
			feature_batch = buckets_feature[bucket_index]
			feature_length_batch = buckets_feature_length[bucket_index]
			sentence_batch = buckets_sentence[bucket_index]
			sentence_length_batch = buckets_sentence_length[bucket_index]
			np.save(os.path.join(data_cache_path, "feature_%d.npy" % bucket_index), feature_batch)
			np.save(os.path.join(data_cache_path, "feature_length_%d.npy" % bucket_index), feature_length_batch)
			np.save(os.path.join(data_cache_path, "sentence_%d.npy" % bucket_index), sentence_batch)
			np.save(os.path.join(data_cache_path, "sentence_length_%d.npy" % bucket_index), sentence_length_batch)

		print(buckets_length)
		raise Exception()


		audio_batch = np.zeros((len(dataset), 3, config.num_mel_filters, max_logmel_length), dtype=np.float32)
		audio_length_batch = np.zeros((len(dataset),), dtype=np.int32)
		sentence_batch = np.zeros((len(dataset), max_sentence_length), dtype=np.int32)
		sentence_length_batch = np.zeros((len(dataset),), dtype=np.int32)


		for idx, data in enumerate(dataset):
			sentence, logmel, delta, delta_delta = data
			assert logmel.shape[1] == delta.shape[1]
			assert delta.shape[1] == delta_delta.shape[1]
			audio_length = logmel.shape[1]
			# cache audio features
			audio_batch[idx, 0, :, :audio_length] = logmel
			audio_batch[idx, 1, :, :audio_length] = delta
			audio_batch[idx, 2, :, :audio_length] = delta_delta
			audio_length_batch[idx] = audio_length
			# cache character ids
			sentence_batch[idx, :len(sentence)] = sentence
			sentence_length_batch[idx] = len(sentence)
			# 平均と標準偏差を計算
			mean_x_batch += np.mean(audio_batch[idx, :, :, :audio_length], axis=2, keepdims=True) / len(dataset)
			stddev_x_batch += np.std(audio_batch[idx, :, :, :audio_length], axis=2, keepdims=True) / len(dataset)

		np.save(feature_filename, audio_batch)
		np.save(feature_length_filename, audio_length_batch)
		np.save(sentence_filename, sentence_batch)
		np.save(sentence_length_filename, sentence_length_batch)
		np.save(mean_filename, mean_x_batch)
		np.save(std_filename, stddev_x_batch)

	# reshape
	mean_x_batch = mean_x_batch[None, ...]
	stddev_x_batch = stddev_x_batch[None, ...]

	return audio_batch, audio_length_batch, sentence_batch, sentence_length_batch, mean_x_batch, stddev_x_batch