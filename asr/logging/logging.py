import codecs
from datetime import datetime

class Level:
	TRACE = 0
	DEBUG = 1
	INFO = 2
	WARN = 3
	ERROR = 4
	FATAL = 5

_level_str = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"]

class Report():
	def __init__(self, filename):
		self.filename = filename

	def __call__(self, body, level=0):
		if isinstance(body, str) is False:
			body = str(body)
		with codecs.open(self.filename, "a", "utf-8") as f:
			f.write(_level_str[level])
			f.write("	")
			f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
			f.write("	")
			f.write(body)
			f.write("\n")
