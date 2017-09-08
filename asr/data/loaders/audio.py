import os
import numpy as np
from ..readers.audio import Reader
from ..processing import Processor
from ...utils import stdout, printb
from .. import iterators

class Loader():
	def __init__(self, wav_directory_list, transcription_directory_list, mean_filename, std_filename, batchsizes, buckets_limit=None, 
		bucket_split_sec=0.5, vocab_token_to_id=None, seed=0, id_blank=0, apply_cmn=False, sampling_rate=16000, frame_width=0.032, 
		frame_shift=0.01, num_mel_filters=40, window_func="hanning", using_delta=True, using_delta_delta=True):

		assert vocab_token_to_id is not None
		assert isinstance(vocab_token_to_id, dict)

		self.batchsizes = batchsizes
		self.token_ids = vocab_token_to_id
		self.id_blank = id_blank
		self.apply_cmn = apply_cmn

		self.processor = Processor(sampling_rate=sampling_rate, frame_width=frame_width, frame_shift=frame_shift, 
			num_mel_filters=num_mel_filters, window_func=window_func, using_delta=using_delta, using_delta_delta=using_delta_delta)

		self.reader = Reader(wav_directory_list=wav_directory_list, transcription_directory_list=transcription_directory_list, 
			buckets_limit=buckets_limit, frame_width=frame_width, bucket_split_sec=bucket_split_sec)

		self.mean = None
		self.std = None

		try:
			if isinstance(mean_filename, str) and isinstance(std_filename, str):
				if os.path.isfile(mean_filename) == False:
					raise Exception()
				if os.path.isfile(std_filename) == False:
					raise Exception()
				self.mean = np.load(mean_filename)[None, ...].astype(np.float32)
				self.std = np.load(std_filename)[None, ...].astype(np.float32)
		except:
			raise Exception("Run preprocess/buckets.py before starting training.")

	def dump(self):
		self.reader.dump()

	def get_total_iterations(self):
		return self.reader.calculate_total_iterations_with_batchsizes(self.batchsizes)

	def get_batch_iterator(self, batchsizes, augmentation=None, gpu=True):
		return iterators.audio.Iterator(self, batchsizes, augmentation, gpu)