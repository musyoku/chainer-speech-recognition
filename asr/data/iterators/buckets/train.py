class Iterator():
	def __init__(self, loader, batchsizes, augmentation=None, gpu=True):
		self.loader = loader
		self.batchsizes = batchsizes
		self.augmentation = None
		self.gpu = gpu
		self.itr = 0
		self.total_itr = loader.get_total_training_iterations()

	def __iter__(self):
		return self

	def __next__(self):
		if self.itr >= self.total_itr:
			raise StopIteration()
		self.itr += 1
		return self.loader.sample_minibatch(self.augmentation, self.gpu)


	def get_total_iterations(self):
		return self.total_itr