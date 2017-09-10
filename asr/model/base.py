import sys, os, json
from ..utils import to_dict, dump_dict, printb, _set

class Configuration():
	def __init__(self):
		self.sampling_rate = 16000
		self.frame_width = 0.032
		self.frame_shift = 0.01
		self.num_mel_filters = 40
		self.window_func = "hanning"
		self.using_delta = True
		self.using_delta_delta = True
		self.bucket_split_sec = 0.5

	def dump(self):
		printb("[Configuration]")
		dump_dict(to_dict(self), 1)

	def save(self, filename, overwrite=False):
		assert self.vocab_size > 0

		if os.path.isfile(filename) and overwrite is False:
			return

		params = to_dict(self)
		
		with open(filename, "w") as f:
			json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

	def load(self, filename):
		if os.path.isfile(filename):
			print("Loading {} ...".format(filename))
			with open(filename, "r") as f:
				try:
					params = json.load(f)
				except Exception as e:
					raise Exception("could not load {}".format(filename))

			_set(self, params)
			return True
		else:
			return None

def configure():
	return Configuration()
