
class Iterator():
	def __init__(self, loader, batchsizes, augmentation=None, gpu=True):
		self.loader = loader
		self.batchsizes = batchsizes
		self.augmentation = None
		self.gpu = gpu
		self.bucket_id = 0
		self.pos = 0
		self.total_itr = loader.get_total_iterations()
		print(self.total_itr)
		raise Exception()

	def __iter__(self):
		return self

	def __next__(self):
		buckets_indices = self.loader.reader.buckets_indices_dev
		if self.bucket_id >= len(buckets_indices):
			raise StopIteration()

		while True:
			bucket_id = self.bucket_id
			piece_id = self.piece_id
			batch = self._next()
			if batch is not None:
				break

			self.piece_id += 1
			self.pos = 0

			if self.piece_id >= len(buckets_indices[bucket_id]):
				self.piece_id = 0
				self.bucket_id += 1

			if self.bucket_id >= len(buckets_indices):
				raise StopIteration()

		audio_features, sentences, max_feature_length, max_sentence_length = self.loader.extract_batch_features(batch, augmentation=self.augmentation)
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.loader.features_to_minibatch(audio_features, sentences, max_feature_length, max_sentence_length, gpu=self.gpu)

		if self.piece_id >= len(buckets_indices[bucket_id]):
			self.piece_id = 0
			self.bucket_id += 1

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_id, piece_id

	def _next(self):
		bucket_id = self.bucket_id
		piece_id = self.piece_id
		buckets_indices = self.loader.reader.buckets_indices_dev

		assert len(buckets_indices) > bucket_id
		assert len(buckets_indices[bucket_id]) > piece_id

		signal_list = self.loader.reader.get_signals_by_bucket_and_piece(bucket_id, piece_id)
		sentence_list = self.loader.reader.get_sentences_by_bucket_and_piece(bucket_id, piece_id)
				
		indices_dev = buckets_indices[bucket_id][piece_id]

		batchsize = self.batchsizes[bucket_id]
		if batchsize > len(indices_dev) - self.pos:
			batchsize = len(indices_dev) - self.pos
		indices = indices_dev[self.pos:self.pos + batchsize]

		if len(indices) == 0:
			return None

		batch = []
		for data_idx in indices:
			signal = signal_list[data_idx]
			sentence = sentence_list[data_idx]
			batch.append((signal, sentence))

		self.pos += batchsize

		if self.pos >= len(indices_dev):
			self.piece_id += 1
			self.pos = 0

		return batch

	def get_total_iterations(self):
		return self.total_itr