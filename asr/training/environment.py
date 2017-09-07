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

def _set(self, d):
	for key in d:
		value = d[key]
		if hasattr(self, key):
			if isinstance(value, dict):
				_set(getattr(self, key), value)
			else:
				setattr(self, key, value)

class Environment(Object):
	def __init__(self, filename, handler):
		self._filename = filename
		self._handler = handler
		signal.signal(signal.SIGUSR1, self.handler)

	def handler(self, _, __):
		self.load()
		self._handler()

	def save(self):
		env = to_dict(self)
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
		_set(self, env)

	def dump(self):
		env = to_dict(self)
		printb("[Environment]")
		_dump(env, 1)
