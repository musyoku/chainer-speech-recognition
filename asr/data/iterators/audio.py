class Iterator():
	def __init__(self, loader, batchsizes, augmentation=None, gpu=True):
		self.loader = loader
		self.batchsizes = batchsizes
		self.augmentation = None
		self.gpu = gpu
		self.bucket_id = 0
		self.pos = 0
		self.total_itr = loader.get_total_iterations()

	def __iter__(self):
		return self

	def __next__(self):
		buckets_signal = self.loader.reader.buckets_signal
		if self.bucket_id >= len(buckets_signal):
			raise StopIteration()

		while True:
			bucket_id = self.bucket_id
			batch = self._next()
			if batch is not None:
				break

			self.pos = 0
			self.bucket_id += 1

			if self.bucket_id >= len(buckets_signal):
				raise StopIteration()

		audio_features, sentences, max_feature_length, max_sentence_length = self.loader.extract_batch_features(batch, augmentation=self.augmentation)
		x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch = self.loader.features_to_minibatch(audio_features, sentences, max_feature_length, max_sentence_length, gpu=self.gpu)

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_id

	def _next(self):
		bucket_id = self.bucket_id
		buckets_signal = self.loader.reader.buckets_signal
		buckets_sentence = self.loader.reader.buckets_sentence

		assert len(buckets_signal) > bucket_id

		batchsize = self.batchsizes[bucket_id]
		if batchsize > len(buckets_signal) - self.pos:
			batchsize = len(buckets_signal) - self.pos

		if batchsize == 0:
			return None

		batch = []
		for i in range(batchsize):
			signal = buckets_signal[bucket_id][self.pos + i]
			sentence = buckets_sentence[bucket_id][self.pos + i]
			batch.append((signal, sentence))

		self.pos += batchsize

		if self.pos >= len(buckets_signal):
			self.pos = 0

		return batch

	def get_total_iterations(self):
		return self.total_itr