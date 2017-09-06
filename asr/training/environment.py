import os, signal

class Environment():
	def __init__(self, filename, handler):
		self._filename = filename
		self._handler = handler
		signal.signal(signal.SIGUSR1, self.handler)

	def handler(self):
		self.load(self._filename)
		self._handler()

	def save():
		env = {}

		for key in dir(self):
			if key.startswith("_") is False:
				setattr(env, key, getattr(self, key))

		with open(self._filename, "w") as f:
			json.dump(env, f, indent=4, sort_keys=True, separators=(',', ': '))

	def load():
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
