import os, signal, json
from ..utils import Object

def _to_dict(obj):
	d = {}
	for key in dir(obj):
		if key.startswith("_") is False:
			attr = getattr(obj, key)
			print(type(attr))
			if isinstance(attr, Object):
				d[key] = _to_dict(attr)
			elif callable(attr):
				pass
			else:
				d[key] = attr
	return d

class Environment(Object):
	def __init__(self, filename, handler):
		self._filename = filename
		self._handler = handler
		signal.signal(signal.SIGUSR1, self.handler)

	def handler(self):
		self.load(self._filename)
		self._handler()

	def save(self):
		env = _to_dict(self)

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
