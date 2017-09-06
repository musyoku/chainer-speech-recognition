import time, sys
from ..utils import printb

def _dump(d, depth):
	for key in d:
		value = d[key]
		if isinstance(value, dict):
			for _ in range(depth):
				sys.stdout.write("	")
			sys.stdout.write("{}:".format(key))
			sys.stdout.write("\n")
			_dump(value, depth + 1)
		else:
			for _ in range(depth):
				sys.stdout.write("	")
			sys.stdout.write(key)
			sys.stdout.write(":	")
			sys.stdout.write(str(value))
			sys.stdout.write("\n")

class Iteration():
	def __init__(self, epochs):
		self.epochs = epochs
		self.itr = 0
		self.epoch_start_time = 0
		self.total_time = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self.itr == self.epochs:
			raise StopIteration()
		self.itr += 1
		self.epoch_start_time = time.time()
		printb("Epoch %d" % self.itr)
		return self.itr

	next = __next__  # Python 2

	def done(self, d):
		elapsed_time = time.time() - self.epoch_start_time
		self.total_time += elapsed_time
		printb("Epoch {} done in {} min - total {} min".format(self.itr, int(elapsed_time / 60), int(self.total_time / 60)))
		_dump(d, 1)
