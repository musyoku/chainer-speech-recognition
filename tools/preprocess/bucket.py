import os, codecs, re, sys, math, chainer, pickle, acoustics, argparse
import numpy as np
import scipy.io.wavfile as wavfile
from chainer import cuda
sys.path.append(os.path.join("..", ".."))
from asr.data.processing import generate_signal_transcription_pairs
from asr.data.readers.buckets import _get_bucket_index
from asr.utils import printr, printc

def main():
	# CSJのwavが入っているディレクトリ
	wav_path_list = [
		"/home/stark/sandbox/CSJ/WAV/core",
		"/home/stark/sandbox/CSJ/WAV/noncore",
	]
	# 変換済みの書き起こしデータ
	# https://github.com/musyoku/csj-preprocesser
	transcription_path_list = [
		"/home/stark/sandbox/CSJ_/core",
		"/home/stark/sandbox/CSJ_/noncore",
	]

	buckets_limit = args.buckets_limit
	buckets_split_sec = args.buckets_split_sec
	num_signals_per_file = args.num_signals_per_file
	dataset_path = args.dataset_path

	buckets_signal = []
	buckets_sentence = []
	buckets_file_indices = []
	max_sentence_length = 0
	max_logmel_length = 0
	current_num_data = 0
	total_min = 0

	def append_bucket(buckets, bucket_id, data):
		if len(buckets) <= bucket_id:
			while len(buckets) <= bucket_id:
				buckets.append([])
		buckets[bucket_id].append(data)

	def add_to_bukcet(signal, sentence, sampling_rate):
		bucket_id = _get_bucket_index(signal, sampling_rate, buckets_split_sec)
		# add signal
		append_bucket(buckets_signal, bucket_id, signal)
		# add sentence
		append_bucket(buckets_sentence, bucket_id, sentence)
		# add file index
		if len(buckets_file_indices) <= bucket_id:
			while len(buckets_file_indices) <= bucket_id:
				buckets_file_indices.append(0)
		# check
		if len(buckets_signal[bucket_id]) >= num_signals_per_file:
			return True, bucket_id
		return False, bucket_id

	def save_bucket(bucket_id):
		if buckets_limit is not None and bucket_id > buckets_limit:
			return False
		file_index = buckets_file_indices[bucket_id]
		num_signals = len(buckets_signal[bucket_id])
		assert num_signals > 0

		with open (os.path.join(dataset_path, "signal", "{}_{}_{}.bucket".format(bucket_id, file_index, num_signals)), "wb") as f:
			pickle.dump(buckets_signal[bucket_id], f)

		num_sentences = len(buckets_sentence[bucket_id])
		assert num_signals == num_sentences
		with open (os.path.join(dataset_path, "sentence", "{}_{}_{}.bucket".format(bucket_id, file_index, num_sentences)), "wb") as f:
			pickle.dump(buckets_sentence[bucket_id], f)
		buckets_signal[bucket_id] = []
		buckets_sentence[bucket_id] = []
		buckets_file_indices[bucket_id] += 1
		return True

	for wav_dir, trn_dir in zip(wav_path_list, transcription_path_list):
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

		for data_id, wav_id in enumerate(sorted(wav_ids)):

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

			printr("Loading {} ({}/{}) ... shape={}, rate={}, min={}, #buckets={}, #data={}".format(wav_filename, data_id + 1, len(wav_fs), audio.shape, sampling_rate, int(duration), len(buckets_signal), current_num_data))

			# 転記の読み込みと音声の切り出し
			signal_transcription_pairs = generate_signal_transcription_pairs(os.path.join(trn_dir, trn_filename), audio, sampling_rate, 512)

			for idx, (signal_sequence, sentence) in enumerate(signal_transcription_pairs):
				# データを確認する場合は書き出し
				# wavfile.write("/home/stark/sandbox/debug/{}.wav".format(sentence), config.sampling_rate, signal_sequence)
				
				write_to_file, bucket_id = add_to_bukcet(signal_sequence, sentence, sampling_rate)
				if write_to_file:
					printr("Writing bucket {} ...".format(bucket_id))
					save_bucket(bucket_id)
					current_num_data += num_signals_per_file


	printr("")
	if data_limit_exceeded == False:
		for bucket_id, bucket in enumerate(buckets_signal):
			if len(bucket) > 0:
				if save_bucket(bucket_id) == False:
					num_unsaved_data = len(buckets_signal[bucket_id])
					print("bucket {} skipped. (#data={})".format(bucket_id, num_unsaved_data))
					current_num_data -= num_unsaved_data
	print("total: {} hour - {} data".format(int(total_min / 60), current_num_data))

def mkdir(d):
	try:
		os.mkdir(d)
	except:
		pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--apply-cmn", "-cmn", default=False, action="store_true")
	parser.add_argument("--buckets-split-sec", "-bsec", type=float, default=0.5)
	parser.add_argument("--buckets-limit", "-limit", type=int, default=20)
	parser.add_argument("--num-signals-per-file", "-nsig", type=int, default=500)
	parser.add_argument("--dataset-path", "-data", type=str, default=None)
	args = parser.parse_args()

	assert args.dataset_path is not None

	mkdir(args.dataset_path)
	mkdir(os.path.join(args.dataset_path, "signal"))
	mkdir(os.path.join(args.dataset_path, "sentence"))
	np.random.seed(0)


	# すべての.wavを読み込み、一定の長さごとに保存
	main()