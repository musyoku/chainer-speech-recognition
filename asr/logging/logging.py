import codecs

class Report():
	def __init__(self, filename):
		self.filename = filename

	def __call__(content):
		with codecs.open(self.filename, "a", "utf-8") as f:
			f.write(content)
			f.write("\n")
