import sys

class Object(object):
	pass

def to_object(d):
	obj = Object()
	_set(obj, d)
	return obj

def _set(obj, d):
	for key in d:
		value = d[key]
		if isinstance(value, dict):
			_set(getattr(obj, key), value)
		else:
			setattr(obj, key, value)

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

def dump_dict(d, depth):
	for key in d:
		value = d[key]
		if isinstance(value, dict):
			for _ in range(depth):
				sys.stdout.write("	")
			sys.stdout.write("{}:".format(key))
			sys.stdout.write("\n")
			dump_dict(value, depth + 1)
		else:
			for _ in range(depth):
				sys.stdout.write("	")
			sys.stdout.write(key)
			sys.stdout.write(":	")
			sys.stdout.write(bold(str(value)))
			sys.stdout.write("\n")

class stdout:
	BOLD = "\033[1m"
	RED = "\033[31m"
	END = "\033[0m"
	CLEAR = "\033[2K"
	MOVE = "\033[1A"
	LEFT = "\033[G"

def bold(string):
	return stdout.BOLD + string + stdout.END

def printb(string):
	print(bold(string))

def printr(string):
	sys.stdout.write("\r" + stdout.CLEAR)
	sys.stdout.write(string)
	sys.stdout.flush()

def printc(string, color):
	code = None

	if color == "red":
		code = stdout.RED

	assert code is not None
	print(code + string + stdout.END)