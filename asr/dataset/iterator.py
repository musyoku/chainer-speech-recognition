from ..utils import printr

class TestBatchIterator():
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

class DevelopmentBatchIterator():
	def __init__(self, dataset, batchsizes, augmentation=None, gpu=True):
		self.dataset = dataset
		self.batchsizes = batchsizes
		self.augmentation = None
		self.gpu = gpu
		self.bucket_idx = 0
		self.group_idx = 0
		self.pos = 0

	def __iter__(self):
		return self

	def __next__(self):
		bucket_idx = self.bucket_idx
		group_idx = self.group_idx
		buckets_indices = self.dataset.reader.buckets_indices_dev

		if bucket_idx >= len(buckets_indices):
			raise StopIteration()

		signal_list = self.dataset.reader.get_signals_by_bucket_and_group(bucket_idx, group_idx)
		sentence_list = self.dataset.reader.get_sentences_by_bucket_and_group(bucket_idx, group_idx)
				
		indices_dev = buckets_indices[bucket_idx][group_idx]

		batchsize = self.batchsizes[bucket_idx]
		batchsize = len(indices_dev) - self.pos if batchsize > len(indices_dev) - self.pos else batchsize
		indices = indices_dev[self.pos:self.pos + batchsize]

		if len(indices) == 0:
			import pdb
			pdb.set_trace()

		assert len(indices) > 0

		batch = []
		for data_idx in indices:
			signal = signal_list[data_idx]
			sentence = sentence_list[data_idx]
			batch.append((signal, sentence))

		audio_features, sentences, max_feature_length, max_sentence_length = self.dataset.extract_batch_features(batch, augmentation=self.augmentation)
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.dataset.features_to_minibatch(audio_features, sentences, max_feature_length, max_sentence_length, gpu=self.gpu)

		self.pos += batchsize
		if self.pos >= len(indices_dev):
			self.group_idx += 1
			self.pos = 0

		if self.group_idx >= len(buckets_indices[bucket_idx]):
			self.group_idx = 0
			self.bucket_idx += 1

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx, group_idx

	next = __next__  # Python 2
		
class TrainingBatchIterator():
	def __init__(self, dataset, batchsizes, augmentation=None, gpu=True):
		self.dataset = dataset
		self.batchsizes = batchsizes
		self.augmentation = None
		self.gpu = gpu
		self.itr = 0
		self.total_itr = dataset.get_total_training_iterations()

	def __iter__(self):
		return self

	def __next__(self):
		if self.itr >= self.total_itr:
			raise StopIteration()
		self.itr += 1
		return self.dataset.sample_minibatch(self.augmentation, self.gpu)

	next = __next__  # Python 2

	def console_log_progress(self):
		printr("iteration {}/{}".format(self.itr, self.total_itr))

	def get_total_iterations(self):
		return self.total_itr