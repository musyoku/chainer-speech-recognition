import re, os, chainer, math
import numpy as np
from ..utils import printb

def _get_bucket_index(signal, sampling_rate=16000, split_sec=0.5):
	divider = sampling_rate * split_sec
	return int(len(signal) // divider)

def _fill(buckets_signal, buckets_sentence, buckets_num_data, buckets_num_updates, bucket_idx, group_idx):
	while len(buckets_signal) <= bucket_idx:
		buckets_signal.append([])
		buckets_sentence.append([])
		buckets_num_data.append([])
		buckets_num_updates.append([])
	while len(buckets_signal[bucket_idx]) <= group_idx:
		buckets_signal[bucket_idx].append(None)
		buckets_sentence[bucket_idx].append(None)
		buckets_num_data[bucket_idx].append(0)
		buckets_num_updates[bucket_idx].append(0)

def generate_buckets_from_raw_data(wav_dir, trn_dir, buckets_limit=None):
	config = chainer.config
	buckets_signal = []
	buckets_sentence = []
	vocab = get_vocab()[0]
	total_min = 0

	def append_bucket(buckets, bucket_idx, data):
		if len(buckets) <= bucket_idx:
			while len(buckets) <= bucket_idx:
				buckets.append([])
		buckets[bucket_idx].append(data)

	def add_to_bukcet(signal, sentence):
		bucket_idx = _get_bucket_index(signal, config.sampling_rate, config.bucket_split_sec)
		if buckets_limit and bucket_idx >= buckets_limit:
			return bucket_idx
		append_bucket(buckets_signal, bucket_idx, signal)
		append_bucket(buckets_sentence, bucket_idx, sentence)
		return bucket_idx

	wav_fs = os.listdir(wav_dir)
	trn_fs = os.listdir(trn_dir)
	wav_ids = set()
	trn_ids = set()
	for filename in wav_fs:
		data_id = re.sub(r"\..+$", "", filename)
		wav_ids.add(data_id)
	for filename in trn_fs:
		data_id = re.sub(r"\..+$", "", filename)
		trn_ids.add(data_id)

	for data_idx, wav_id in enumerate(wav_ids):
		if wav_id not in trn_ids:
			sys.stdout.write("\r")
			sys.stdout.write(stdout.CLEAR)
			print("%s.trn not found" % wav_id)
			continue

		wav_filename = "%s.wav" % wav_id
		trn_filename = "%s.trn" % wav_id

		# wavの読み込み
		try:
			sampling_rate, audio = wavfile.read(os.path.join(wav_dir, wav_filename))
		except KeyboardInterrupt:
			exit()
		except Exception as e:
			sys.stdout.write("\r")
			sys.stdout.write(stdout.CLEAR)
			print("Failed to read {} ({})".format(wav_filename, str(e)))
			continue

		duration = audio.size / sampling_rate / 60
		total_min += duration

		sys.stdout.write("\r")
		sys.stdout.write(stdout.CLEAR)
		sys.stdout.write("Reading {} ({}/{}) ... shape={}, rate={}, min={}".format(wav_filename, data_idx + 1, len(wav_fs), audio.shape, sampling_rate, int(duration)))
		sys.stdout.flush()

		# 転記の読み込みと音声の切り出し
		batch = generate_signal_transcription_pairs(os.path.join(trn_dir, trn_filename), audio, sampling_rate, vocab)

		# 信号長と転記文字列長の不自然な部分を検出
		num_points_per_character = 0	# 1文字あたりの信号の数
		for signal, sentence in batch:
			num_points_per_character += len(signal) / len(sentence)
		num_points_per_character /= len(signal)

		accept_rate = 0.7	# ズレの割合がこれ以下なら教師データに誤りが含まれている可能性があるので目視で確認すべき

		for idx, (signal, sentence) in enumerate(batch):
			error = abs(len(signal) - num_points_per_character * len(sentence))
			rate = error / len(signal)
			if rate < accept_rate:
				raise Exception(len(signal), len(sentence), num_points_per_character, rate, trn_filename, idx + 1)
			
			add_to_bukcet(signal, sentence)

	sys.stdout.write("\r")
	sys.stdout.write(stdout.CLEAR)
	printb("bucket	#data	sec")
	total = 0
	for bucket_idx, signals in enumerate(buckets_signal):
		if buckets_limit and bucket_idx >= buckets_limit:
			break
		num_data = len(signals)
		total += num_data
		print("{}	{:>4}	{:>6.3f}".format(bucket_idx + 1, num_data, config.bucket_split_sec * (bucket_idx + 1)))
	print("total	{:>4}		{} hour".format(total, int(total_min / 60)))

	return buckets_signal, buckets_sentence

class BucketsReader():
	def __init__(self, data_path, buckets_limit=None, buckets_cache_size=200, dev_split=0.01, seed=0):
		self.buckets_cache_size = buckets_cache_size
		self.dev_split = dev_split
		self.data_path = data_path
		self.buckets_limit = buckets_limit

		signal_path = os.path.join(data_path, "signal")
		sentence_path = os.path.join(data_path, "sentence")
		signal_files = os.listdir(signal_path)
		sentence_files = os.listdir(sentence_path)

		try:
			if len(signal_files) == 0:
				raise Exception()
			if len(sentence_files) == 0:
				raise Exception()
			if len(signal_files) != len(sentence_files):
				raise Exception()
		except:
			raise Exception("Run preprocess/buckets.py before starting training.")

		buckets_signal = []
		buckets_sentence = []
		buckets_num_data = []
		buckets_num_updates = []
		for filename in signal_files:
			pattern = r"([0-9]+)_([0-9]+)_([0-9]+)\.bucket"
			m = re.match(pattern , filename)
			if m:
				bucket_idx = int(m.group(1))
				group_idx = int(m.group(2))
				num_data = int(m.group(3))
				_fill(buckets_signal, buckets_sentence, buckets_num_data, buckets_num_updates, bucket_idx, group_idx)
				buckets_num_data[bucket_idx][group_idx] = num_data

		if buckets_limit is not None:
			buckets_signal = buckets_signal[:buckets_limit]
			buckets_sentence = buckets_sentence[:buckets_limit]
			buckets_num_data = buckets_num_data[:buckets_limit]

		buckets_num_group = []
		for bucket in buckets_signal:
			buckets_num_group.append(len(bucket))
		total_groups = sum(buckets_num_group)
		total_buckets = len(buckets_signal)

		np.random.seed(seed)
		buckets_indices_train = []
		buckets_indices_dev = []
		for bucket_idx in range(total_buckets):
			num_groups = buckets_num_group[bucket_idx]
			indices_train = []
			indices_dev = []
			for group_idx in range(num_groups):
				num_data = buckets_num_data[bucket_idx][group_idx]
				indices = np.arange(num_data)
				np.random.shuffle(indices)
				num_dev = int(num_data * dev_split)
				indices_train.append(indices[num_dev:])
				if num_dev > 0:
					indices_dev.append(indices[:num_dev])
			buckets_indices_train.append(indices_train)
			buckets_indices_dev.append(indices_dev)

		self.buckets_signal = buckets_signal
		self.buckets_sentence = buckets_sentence
		self.buckets_num_group = buckets_num_group
		self.buckets_num_data = buckets_num_data
		self.buckets_num_updates = buckets_num_updates
		self.buckets_indices_train = buckets_indices_train
		self.buckets_indices_dev = buckets_indices_dev

		self.total_groups = total_groups
		self.total_buckets = total_buckets
		self.bucket_distribution = np.asarray(buckets_num_group) / total_groups
		self.cached_indices = []

	def get_signals_by_bucket_and_group(self, bucket_idx, group_idx):
		num_data = self.buckets_num_data[bucket_idx][group_idx]
		signal_list = self.buckets_signal[bucket_idx][group_idx]
		if signal_list is None:
			with open(os.path.join(self.data_path, "signal", "{}_{}_{}.bucket".format(bucket_idx, group_idx, num_data)), "rb") as f:
				signal_list = pickle.load(f)
				self.buckets_signal[bucket_idx][group_idx] = signal_list

		# 一定以上はメモリ解放
		if self.buckets_cache_size > 0:
			self.cached_indices.append((bucket_idx, group_idx))
			if len(self.cached_indices) > self.buckets_cache_size:
				_bucket_idx, _group_idx = self.cached_indices.pop(0)
				self.buckets_signal[_bucket_idx][_group_idx] = None
				self.buckets_sentence[bucket_idx][group_idx] = None

		return signal_list

	def get_sentences_by_bucket_and_group(self, bucket_idx, group_idx):
		num_data = self.buckets_num_data[bucket_idx][group_idx]
		sentence_list = self.buckets_sentence[bucket_idx][group_idx]
		if sentence_list is None:
			with open (os.path.join(self.data_path, "sentence", "{}_{}_{}.bucket".format(bucket_idx, group_idx, num_data)), "rb") as f:
				sentence_list = pickle.load(f)
				self.buckets_sentence[bucket_idx][group_idx] = sentence_list

		return sentence_list

	def increment_num_updates(self, bucket_idx, group_idx):
		self.buckets_num_updates[bucket_idx][group_idx] += 1

	def sample_minibatch(self, batchsizes):
		bucket_idx = np.random.choice(np.arange(len(self.buckets_signal)), size=1, p=self.bucket_distribution)[0]
		group_idx = np.random.choice(np.arange(self.buckets_num_group[bucket_idx]), size=1)[0]

		signal_list = self.get_signals_by_bucket_and_group(bucket_idx, group_idx)
		sentence_list = self.get_sentences_by_bucket_and_group(bucket_idx, group_idx)

		self.increment_num_updates(bucket_idx, group_idx)
	
		indices = self.buckets_indices_train[bucket_idx][group_idx]
		np.random.shuffle(indices)

		batchsize = batchsizes[bucket_idx]
		batchsize = len(indices) if batchsize > len(indices) else batchsize
		indices = indices[:batchsize]

		batch = []
		for data_idx in indices:
			signal = signal_list[data_idx]
			sentence = sentence_list[data_idx]
			batch.append((signal, sentences))

		return batch

	def dump_num_updates(self):
		for bucket_idx in range(len(self.buckets_signal)):
			printb("bucket " + str(bucket_idx))
			buckets = self.buckets_num_updates[bucket_idx]
			print(buckets)
			print(sum(buckets) / len(buckets))

	def dump_information(self):
		printb("bucket	#train	#dev	sec")
		total_train = 0
		total_dev = 0
		config = chainer.config
		for bucket_idx, (indices_group_train, indices_group_dev) in enumerate(zip(self.buckets_indices_train, self.buckets_indices_dev)):
			if self.buckets_limit is not None and bucket_idx >= self.buckets_limit:
				break
			num_train = 0
			num_dev = 0
			for indices_train in indices_group_train:
				total_train += len(indices_train)
				num_train += len(indices_train)
			for indices_dev in indices_group_dev:
				total_dev += len(indices_dev)
				num_dev += len(indices_dev)
			print("{}	{:>6}	{:>4}	{:>6.3f}".format(bucket_idx + 1, num_train, num_dev, config.bucket_split_sec * (bucket_idx + 1)))
		print("total	{:>6}	{:>4}".format(total_train, total_dev))

	def calculate_total_training_iterations_with_batchsizes(self, batchsizes):
		num_buckets = len(self.buckets_signal)
		batchsizes = batchsizes[:num_buckets]
		itr = 0
		for indices_group_train, batchsize in zip(self.buckets_indices_train, batchsizes):
			for indices_train in indices_group_train:
				itr += int(math.ceil(len(indices_train) / batchsize))
		return itr