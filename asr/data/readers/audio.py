import re, os, chainer, math, pickle
import numpy as np
import scipy.io.wavfile as wavfile
from ...utils import printb, printr, printc
from ..processing import generate_signal_transcription_pairs

def _get_bucket_index(signal, sampling_rate=16000, split_sec=0.5):
	divider = sampling_rate * split_sec
	return int(len(signal) // divider)

class Reader():
	def __init__(self, wav_directory_list, transcription_directory_list, buckets_limit=None, frame_width=0.032, bucket_split_sec=0.5):
		self.wav_directory_list = wav_directory_list
		self.transcription_directory_list = transcription_directory_list
		self.buckets_limit = buckets_limit
		self.frame_width = frame_width
		self.bucket_split_sec = bucket_split_sec
		self.buckets_signal = []
		self.buckets_sentence = []
		self.total_min = 0

		for wav_dir, trn_dir in zip(wav_directory_list, transcription_directory_list):
			buckets_signal, buckets_sentence, total_min = self.read(wav_dir, trn_dir)
			self.buckets_signal += buckets_signal
			self.buckets_sentence += buckets_sentence
			self.total_min += total_min

	def read(self, wav_dir, trn_dir):
		buckets_signal = []
		buckets_sentence = []
		total_min = 0

		def append_bucket(buckets, bucket_id, data):
			if len(buckets) <= bucket_id:
				while len(buckets) <= bucket_id:
					buckets.append([])
			buckets[bucket_id].append(data)

		def add_to_bukcet(signal, sentence, sampling_rate):
			bucket_id = _get_bucket_index(signal, sampling_rate, self.bucket_split_sec)
			if self.buckets_limit and bucket_id >= self.buckets_limit:
				return bucket_id
			append_bucket(buckets_signal, bucket_id, signal)
			append_bucket(buckets_sentence, bucket_id, sentence)
			return bucket_id

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

		for data_id, wav_id in enumerate(wav_ids):
			if wav_id not in trn_ids:
				printr("%s.trn not found" % wav_id)
				continue

			wav_filename = "%s.wav" % wav_id
			trn_filename = "%s.trn" % wav_id

			# wavの読み込み
			try:
				sampling_rate, audio = wavfile.read(os.path.join(wav_dir, wav_filename))
			except KeyboardInterrupt:
				exit()
			except Exception as e:
				printr("")
				printc("Failed to read {} ({})".format(wav_filename, str(e)), color="red")
				continue

			duration = audio.size / sampling_rate / 60
			total_min += duration

			printr("Reading {} ({}/{}) ... shape={}, rate={}, min={}".format(wav_filename, data_id + 1, len(wav_fs), audio.shape, sampling_rate, int(duration)))

			# 転記の読み込みと音声の切り出し
			batch = generate_signal_transcription_pairs(os.path.join(trn_dir, trn_filename), audio, sampling_rate, int(sampling_rate * self.frame_width))

			for idx, (signal, sentence) in enumerate(batch):
				add_to_bukcet(signal, sentence, sampling_rate)

		return buckets_signal, buckets_sentence, total_min

	def calculate_total_iterations_with_batchsizes(self, batchsizes):
		num_buckets = len(self.buckets_signal)
		batchsizes = batchsizes[:num_buckets]
		itr = 0
		for buckets, batchsize in zip(self.buckets_signal, batchsizes):
			itr += int(math.ceil(len(buckets) / batchsize))
		return itr

	def dump(self):
		printr("")
		printb("bucket	#data	sec")
		total = 0
		for bucket_id, signals in enumerate(self.buckets_signal):
			if self.buckets_limit and bucket_id >= self.buckets_limit:
				break
			num_data = len(signals)
			total += num_data
			print("{}	{:>4}	{:>6.3f}".format(bucket_id + 1, num_data, self.bucket_split_sec * (bucket_id + 1)))
		print("total	{:>4}		{} hour".format(total, int(self.total_min / 60)))

	def get_num_buckets(self):
		return len(self.buckets_signal)