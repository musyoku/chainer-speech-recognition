import os, signal, json, sys
from ..utils import Object, to_dict, printb

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

class Environment(Object):
	def __init__(self, filename, handler):
		self._filename = filename
		self._handler = handler
		signal.signal(signal.SIGUSR1, self.handler)

	def handler(self):
		self.load(self._filename)
		self._handler()

	def save(self):
		env = to_dict(self)

		print(env)
		with open(self._filename, "w") as f:
			json.dump(env, f, indent=4, sort_keys=True, separators=(',', ': '))

	def load(self):
		env = None
		if os.path.isfile(self._filename):
			with open(self._filename, "r") as f:
				try:
					env = json.load(f)
				except:
					pass
		assert env is not None, "could not load {}".format(self._filename)

		for key in dir(env):
			if key.startswith("_") is False:
				setattr(self, key, getattr(env, key))

	def dump(self):
		env = to_dict(self)
		printb("[Environment]")
		_dump(env, 0)
