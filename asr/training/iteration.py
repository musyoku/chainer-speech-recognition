import time, sys
from ..utils import printb, stdout

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
		self.current_epoch = 0
		self.current_epoch_start_time = 0
		self.start_time = 0
		self.epoch = None

	def __iter__(self):
		return self

	def __next__(self):
		if self.current_epoch == self.epochs:
			raise StopIteration()

		if self.start_time == 0:
			self.start_time = time.time()

		self.current_epoch += 1
		self.current_epoch_start_time = time.time()
		printb("Epoch %d" % self.current_epoch)
		return self.current_epoch

	def log_progress(self, string):
		printr(string)

	def console_log(self, d):
		sys.stdout.write(stdout.CLEAR)
		sys.stdout.write(stdout.MOVE)
		sys.stdout.write(stdout.CLEAR)
		sys.stdout.write(stdout.LEFT)
		total_time = time.time() - self.start_time
		elapsed_time = time.time() - self.current_epoch_start_time
		printb("Epoch {} done in {} min - total {} min".format(self.current_epoch, int(elapsed_time / 60), int(total_time / 60)))
		_dump(d, 1)

	next = __next__  # Python 2