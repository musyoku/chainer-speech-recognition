class Loader():
	def __init__(self, data_path, batchsizes_train, batchsizes_dev=None, buckets_limit=None, buckets_cache_size=200, 
		vocab_token_to_id=None, dev_split=0.01, seed=0, id_blank=0, apply_cmn=False, global_normalization=True, 
		sampling_rate=16000, frame_width=0.032, frame_shift=0.01, num_mel_filters=40, window_func="hanning",
		using_delta=True, using_delta_delta=True):

		self.processor = Processor(sampling_rate=sampling_rate, frame_width=frame_width, frame_shift=frame_shift, 
			num_mel_filters=num_mel_filters, window_func=window_func, using_delta=using_delta, using_delta_delta=using_delta_delta)

		self.dataset = Dataset(data_path=data_path, batchsizes_train=batchsizes_train, batchsizes_dev=batchsizes_dev, 
			buckets_limit=buckets_limit, token_ids=vocab_token_to_id, id_blank=id_blank, buckets_cache_size=buckets_cache_size, 
			apply_cmn=apply_cmn)

	def sample_minibatch(self, augmentation=None, gpu=True):
		return self.dataset.sample_minibatch(self.processor, augmentation, gpu)

	def set_batchsizes_train(self, batchsizes):
		self.dataset.set_batchsizes_train(batchsizes)

	def set_batchsizes_dev(self, batchsizes):
		self.dataset.set_batchsizes_dev(batchsizes)

	def get_training_batch_iterator(self, batchsizes, augmentation=None, gpu=True):
		return self.dataset.get_training_batch_iterator(batchsizes, augmentation, gpu)

	def get_development_batch_iterator(self, batchsizes, augmentation=None, gpu=True):
		return self.dataset.get_development_batch_iterator(batchsizes, augmentation, gpu)

	def get_total_training_iterations(self):
		return self.dataset.get_total_training_iterations()

	def get_total_dev_iterations(self):
		return self.dataset.get_total_dev_iterations()

	def get_statistics(self):
		return self.dataset.get_statistics()

	def dump(self):
		self.dataset.dump()
