import re, os, chainer, math, pickle
import numpy as np
import scipy.io.wavfile as wavfile
from ...utils import printb, printr, printc
from ..processing import generate_signal_transcription_pairs

def _get_bucket_index(signal, sampling_rate=16000, split_sec=0.5):
	divider = sampling_rate * split_sec
	return int(len(signal) // divider)

def _fill(buckets_signal, buckets_sentence, buckets_num_data, buckets_num_updates, bucket_id, piece_id):
	while len(buckets_signal) <= bucket_id:
		buckets_signal.append([])
		buckets_sentence.append([])
		buckets_num_data.append([])
		buckets_num_updates.append([])
	assert len(buckets_signal) == len(buckets_sentence)
	assert len(buckets_sentence) == len(buckets_num_data)
	assert len(buckets_num_data) == len(buckets_num_updates)
	while len(buckets_signal[bucket_id]) <= piece_id:
		buckets_signal[bucket_id].append(None)
		buckets_sentence[bucket_id].append(None)
		buckets_num_data[bucket_id].append(0)
		buckets_num_updates[bucket_id].append(0)

class Reader():
	def __init__(self, data_path, buckets_limit=None, buckets_cache_size=200, dev_split=0.01, seed=0, sampling_rate=16000, bucket_split_sec=0.5):
		self.buckets_cache_size = buckets_cache_size
		self.dev_split = dev_split
		self.data_path = data_path
		self.buckets_limit = buckets_limit
		self.bucket_split_sec = bucket_split_sec

		signal_path = os.path.join(data_path, "signal")
		sentence_path = os.path.join(data_path, "sentence")
		signal_files = os.listdir(signal_path)
		sentence_files = os.listdir(sentence_path)

		try:
			if len(signal_files) == 0:
				raise Exception()
			if len(sentence_files) == 0:
				raise Exception()
			if len(signal_files) != len(sentence_files):
				raise Exception()
		except:
			raise Exception("Run preprocess/buckets.py before starting training.")

		buckets_signal = []
		buckets_sentence = []
		buckets_num_data = []
		buckets_num_updates = []
		for filename in signal_files:
			pattern = r"([0-9]+)_([0-9]+)_([0-9]+)\.bucket"
			m = re.match(pattern , filename)
			if m:
				bucket_id = int(m.group(1))
				piece_id = int(m.group(2))
				num_data = int(m.group(3))
				_fill(buckets_signal, buckets_sentence, buckets_num_data, buckets_num_updates, bucket_id, piece_id)
				buckets_num_data[bucket_id][piece_id] = num_data

		if buckets_limit is not None:
			buckets_signal = buckets_signal[:buckets_limit]
			buckets_sentence = buckets_sentence[:buckets_limit]
			buckets_num_data = buckets_num_data[:buckets_limit]
			buckets_num_updates = buckets_num_updates[:buckets_limit]

		buckets_num_pieces = []
		for bucket in buckets_signal:
			buckets_num_pieces.append(len(bucket))
		total_pieces = sum(buckets_num_pieces)
		total_buckets = len(buckets_signal)

		np.random.seed(seed)
		buckets_indices_train = []
		buckets_indices_dev = []
		for bucket_id in range(total_buckets):
			num_pieces = buckets_num_pieces[bucket_id]
			indices_train = []
			indices_dev = []
			for piece_id in range(num_pieces):
				num_data = buckets_num_data[bucket_id][piece_id]
				indices = np.arange(num_data)
				np.random.shuffle(indices)
				num_dev = int(num_data * dev_split)
				indices_train.append(indices[num_dev:])
				if num_dev == 0:
					indices_dev.append([])
				else:
					indices_dev.append(indices[:num_dev])
			assert len(indices_train) == len(indices_dev)
			buckets_indices_train.append(indices_train)
			buckets_indices_dev.append(indices_dev)

		self.buckets_signal = buckets_signal
		self.buckets_sentence = buckets_sentence
		self.buckets_num_pieces = buckets_num_pieces
		self.buckets_num_data = buckets_num_data
		self.buckets_num_updates = buckets_num_updates
		self.buckets_indices_train = buckets_indices_train
		self.buckets_indices_dev = buckets_indices_dev

		self.total_pieces = total_pieces
		self.total_buckets = total_buckets
		self.bucket_distribution = np.asarray(buckets_num_pieces) / total_pieces
		self.cached_indices = []

	def get_signals_by_bucket_and_piece(self, bucket_id, piece_id):
		num_data = self.buckets_num_data[bucket_id][piece_id]
		signal_list = self.buckets_signal[bucket_id][piece_id]
		if signal_list is None:
			with open(os.path.join(self.data_path, "signal", "{}_{}_{}.bucket".format(bucket_id, piece_id, num_data)), "rb") as f:
				signal_list = pickle.load(f)
				self.buckets_signal[bucket_id][piece_id] = signal_list

		# 一定以上はメモリ解放
		if self.buckets_cache_size > 0:
			self.cached_indices.append((bucket_id, piece_id))
			if len(self.cached_indices) > self.buckets_cache_size:
				_bucket_id, _piece_id = self.cached_indices.pop(0)
				self.buckets_signal[_bucket_id][_piece_id] = None
				self.buckets_sentence[bucket_id][piece_id] = None

		return signal_list

	def get_sentences_by_bucket_and_piece(self, bucket_id, piece_id):
		num_data = self.buckets_num_data[bucket_id][piece_id]
		sentence_list = self.buckets_sentence[bucket_id][piece_id]
		if sentence_list is None:
			with open (os.path.join(self.data_path, "sentence", "{}_{}_{}.bucket".format(bucket_id, piece_id, num_data)), "rb") as f:
				sentence_list = pickle.load(f)
				self.buckets_sentence[bucket_id][piece_id] = sentence_list

		return sentence_list

	def increment_num_updates(self, bucket_id, piece_id):
		self.buckets_num_updates[bucket_id][piece_id] += 1

	def sample_minibatch(self, batchsizes):
		bucket_id = np.random.choice(np.arange(len(self.buckets_signal)), size=1, p=self.bucket_distribution)[0]
		piece_id = np.random.choice(np.arange(self.buckets_num_pieces[bucket_id]), size=1)[0]

		signal_list = self.get_signals_by_bucket_and_piece(bucket_id, piece_id)
		sentence_list = self.get_sentences_by_bucket_and_piece(bucket_id, piece_id)

		self.increment_num_updates(bucket_id, piece_id)
	
		indices = self.buckets_indices_train[bucket_id][piece_id]
		np.random.shuffle(indices)

		batchsize = batchsizes[bucket_id]
		batchsize = len(indices) if batchsize > len(indices) else batchsize
		indices = indices[:batchsize]

		batch = []
		for data_id in indices:
			signal = signal_list[data_id]
			sentence = sentence_list[data_id]
			batch.append((signal, sentence))

		return batch, bucket_id, piece_id

	def get_statistics(self):
		content = ""
		for bucket_id in range(len(self.buckets_signal)):
			content += "bucket {}\n".format(str(bucket_id + 1))
			buckets = self.buckets_num_updates[bucket_id]
			content += "{}\n".format(str(buckets))
			content += "{}\n".format(sum(buckets) / len(buckets))
		return content

	def dump(self):
		printb("	bucket	#train	#dev	sec")
		total_train = 0
		total_dev = 0
		for bucket_id, (indices_piece_train, indices_piece_dev) in enumerate(zip(self.buckets_indices_train, self.buckets_indices_dev)):
			if self.buckets_limit is not None and bucket_id >= self.buckets_limit:
				break
			num_train = 0
			num_dev = 0
			for indices_train in indices_piece_train:
				total_train += len(indices_train)
				num_train += len(indices_train)
			for indices_dev in indices_piece_dev:
				total_dev += len(indices_dev)
				num_dev += len(indices_dev)
			print("	{}	{:>6}	{:>4}	{:>6.3f}".format(bucket_id + 1, num_train, num_dev, self.bucket_split_sec * (bucket_id + 1)))
		print("	total	{:>6}	{:>4}".format(total_train, total_dev))

	def calculate_total_training_iterations_with_batchsizes(self, batchsizes):
		num_buckets = len(self.buckets_signal)
		batchsizes = batchsizes[:num_buckets]
		itr = 0
		for indices_piece_train, batchsize in zip(self.buckets_indices_train, batchsizes):
			for indices_train in indices_piece_train:
				itr += int(math.ceil(len(indices_train) / batchsize))
		return itr

	def calculate_total_dev_iterations_with_batchsizes(self, batchsizes):
		num_buckets = len(self.buckets_signal)
		batchsizes = batchsizes[:num_buckets]
		itr = 0
		for indices_piece_dev, batchsize in zip(self.buckets_indices_dev, batchsizes):
			for indices_train in indices_piece_dev:
				itr += int(math.ceil(len(indices_train) / batchsize))
		return itr

	def get_num_buckets(self):
		return len(self.buckets_signal)
