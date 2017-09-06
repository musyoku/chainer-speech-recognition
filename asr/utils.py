import sys

class Object(object):
	pass

def to_dict(obj):
	d = {}
	for key in dir(obj):
		if key.startswith("_") is False:
			attr = getattr(obj, key)
			if isinstance(attr, Object):
				d[key] = to_dict(attr)
			elif callable(attr):
				pass
			else:
				d[key] = attr
	return d

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"
	MOVE = "\033[1A"
	LEFT = "\033[G"

def printb(string):
	print(stdout.BOLD + string + stdout.END)

def printr(string):
	sys.stdout.write("\r" + stdout.CLEAR)
	sys.stdout.write(string)
	sys.stdout.flush()