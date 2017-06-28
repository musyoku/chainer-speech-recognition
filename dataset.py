# coding: utf-8
from __future__ import division
from __future__ import print_function
import os, codecs, re, sys
import chainer
import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import logfbank
from python_speech_features import fbank

BUCKET_THRESHOLD = 32

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

def get_vocab():
	characters = [
		u"_",	# blank
		u"あ",u"い",u"う",u"え",u"お",
		u"か",u"き",u"く",u"け",u"こ",
		u"さ",u"し",u"す",u"せ",u"そ",
		u"た",u"ち",u"つ",u"て",u"と",
		u"な",u"に",u"ぬ",u"ね",u"の",
		u"は",u"ひ",u"ふ",u"へ",u"ほ",
		u"ま",u"み",u"む",u"め",u"も",
		u"や",u"ゆ",u"よ",
		u"ら",u"り",u"る",u"れ",u"ろ",
		u"わ",u"を",u"ん",
		u"が",u"ぎ",u"ぐ",u"げ",u"ご",
		u"ざ",u"じ",u"ず",u"ぜ",u"ぞ",
		u"だ",u"ぢ",u"づ",u"で",u"ど",
		u"ば",u"び",u"ぶ",u"べ",u"ぼ",
		u"ぱ",u"ぴ",u"ぷ",u"ぺ",u"ぽ",
		u"ぁ",u"ぃ",u"ぅ",u"ぇ",u"ぉ",
		u"ゃ",u"ゅ",u"ょ",
		u"っ",
		u"ー",
	]

	vocab = {}
	for char in characters:
		vocab[char] = len(vocab)

	vocab_inv = {}
	for char, char_id in vocab.items():
		vocab_inv[char_id] = char

	id_blank = 0

	return vocab, vocab_inv, id_blank

def get_minibatch(bucket, dataset, batchsize, id_blank):
	assert len(bucket) >= batchsize
	config = chainer.config
	indices = bucket[:batchsize]
	max_x_width = 0
	max_t_width = 0

	for idx in indices:
		data = dataset[idx]
		signal, sentence, logmel, delta, delta_delta = data
		if logmel is None:
			logmel, delta, delta_delta = extract_features(signal, config.sampling_rate, config.num_fft, config.frame_width, config.frame_shift, config.num_mel_filters, config.window_func, config.using_delta, config.using_delta_delta)
			dataset[idx] = signal, sentence, logmel, delta, delta_delta
		if len(sentence) > max_t_width:
			max_t_width = len(sentence)
		if logmel.shape[1] > max_x_width:
			max_x_width = logmel.shape[1]

	x_batch = np.zeros((batchsize, 3, config.num_mel_filters, max_x_width), dtype=np.float32)
	t_batch = np.full((batchsize, max_t_width), id_blank, dtype=np.int32)
	x_valid_length = []
	t_valid_length = []

	for batch_idx, data_idx in enumerate(indices):
		data = dataset[data_idx]
		signal, sentence, logmel, delta, delta_delta = data
		x_length = logmel.shape[1]
		t_length = len(sentence)

		# x
		x_batch[batch_idx, 0, :, :logmel.shape[1]] = logmel
		x_batch[batch_idx, 1, :, :delta.shape[1]] = delta
		x_batch[batch_idx, 2, :, :delta_delta.shape[1]] = delta_delta
		x_valid_length.append(x_length)

		# CTCが適用可能かチェック
		num_trans_same_label = np.count_nonzero(sentence == np.roll(sentence, 1))
		required_length = t_length * 2 + 1 + num_trans_same_label
		if x_length < required_length:
			possibole_t_length = (x_length - num_trans_same_label - 1) // 2
			sentence = sentence[:possibole_t_length]
			t_length = len(sentence)

		# t
		for pos, char_id in enumerate(sentence):
			t_batch[batch_idx, pos] = char_id
		t_valid_length.append(t_length)

	return x_batch, x_valid_length, t_batch, t_valid_length

def extract_features(signal, sampling_rate=16000, num_fft=512, frame_width=0.032, frame_shift=0.01, num_mel_filters=40, window_func=lambda x:np.hanning(x), using_delta=True, using_delta_delta=True):
	# メルフィルタバンク出力の対数を計算
	logmel, energy = fbank(signal, sampling_rate, nfft=num_fft, winlen=frame_width, winstep=frame_shift, nfilt=num_mel_filters, winfunc=window_func)
	logmel = np.log(logmel)

	# ΔとΔΔを計算
	delta = (np.roll(logmel, -1, axis=0) - logmel) / 2 if using_delta else None
	delta_delta = (np.roll(delta, -1, axis=0) - delta) / 2 if using_delta_delta else None

	# 不要な部分を削除
	# ΔΔまで計算すると末尾の2つは正しくない値になる
	logmel = logmel[:-2].T
	delta = delta[:-2].T if using_delta else None
	delta_delta = delta_delta[:-2].T if using_delta_delta else None

	return logmel, delta, delta_delta

def load_audio_features_and_transcriptions(wav_paths, transcription_paths, buckets_limit, data_limit):
	assert len(wav_paths) > 0
	assert len(transcription_paths) > 0

	config = chainer.config
	vocab = get_vocab()[0]
	dataset = []
	max_sentence_length = 0
	max_logmel_length = 0

	for wav_dir, trn_dir in zip(wav_paths, transcription_paths):
		if data_limit is not None and len(dataset) > data_limit:
			break
		wav_fs = os.listdir(wav_dir)
		trn_fs = os.listdir(trn_dir)
		wav_fs.sort()
		trn_fs.sort()
		assert len(wav_fs) == len(trn_fs)

		for data_idx, (wav_filename, trn_filename) in enumerate(zip(wav_fs, trn_fs)):
			if data_limit is not None and len(dataset) > data_limit:
				break
			wav_id = re.sub(r"\..+$", "", wav_filename)
			trn_id = re.sub(r"\..+$", "", trn_filename)
			if wav_id != trn_id:
				raise Exception("{} != {}".format(wav_id, trn_id))

			# wavの読み込み
			try:
				sampling_rate, audio = wavfile.read(os.path.join(wav_dir, wav_filename))
			except KeyboardInterrupt:
				exit()
			except:
				print("{} をスキップしました（読み込みできません）".format(wav_filename))
				continue

			sys.stdout.write("\r")
			sys.stdout.write(stdout.CLEAR)
			sys.stdout.write("loading {} ({}/{}) ... shape={}, rate={}, min={}".format(wav_filename, data_idx + 1, len(wav_fs), audio.shape, sampling_rate, int(audio.size / sampling_rate / 60)))
			sys.stdout.flush()

			# 転記の読み込み
			batch = []
			with codecs.open(os.path.join(trn_dir, trn_filename), "r", "utf-8") as f:
				for data in f:
					period_str, channel, sentence = data.split(":")
					period = period_str.split("-")
					start_sec, end_sec = float(period[0]), float(period[1])
					start_frame = int(start_sec * sampling_rate)
					end_frame = int(end_sec * sampling_rate)

					assert start_frame <= len(audio)
					assert end_frame <= len(audio)

					signal = audio[start_frame:end_frame]

					assert len(signal) == end_frame - start_frame

					# channelに従って選択
					if signal.ndim == 2:
						if channel == "S":	# 両方に含まれる場合
							signal = signal[:, 0]
						elif channel == "L":
							signal = signal[:, 0]
						elif channel == "R":
							signal = signal[:, 1]
						else:
							raise Exception()

					# 文字IDに変換
					char_id_sequence = []
					sentence = sentence.strip()
					for char in sentence:
						if char not in vocab:
							continue
						char_id = vocab[char]
						char_id_sequence.append(char_id)

					batch.append((signal, char_id_sequence))

			# 信号長と転記文字列長の不自然な部分を検出
			num_points_per_character = 0	# 1文字あたりの信号の数
			for signal, char_id_sequence in batch:
				num_points_per_character += len(signal) / len(char_id_sequence)
			num_points_per_character /= len(signal)

			accept_rate = 0.4	# ズレの割合がこれ以下なら教師データに誤りが含まれている可能性があるので目視で確認すべき
			if trn_filename == "M03F0017.trn":	# CSJのこのファイルだけ異常な早口がある
				accept_rate = 0.05
			for idx, (signal, char_id_sequence) in enumerate(batch):
				error = abs(len(signal) - num_points_per_character * len(char_id_sequence))
				rate = error / len(signal)
				if rate < accept_rate:
					raise Exception(len(signal), len(char_id_sequence), num_points_per_character, rate, trn_filename, idx + 1)
			
				logmel, delta, delta_delta = extract_features(signal, config.sampling_rate, config.num_fft, config.frame_width, config.frame_shift, config.num_mel_filters, config.window_func, config.using_delta, config.using_delta_delta)
				bucket_idx = get_bucket_idx(logmel.shape[1])
				if bucket_idx < buckets_limit:
					dataset.append((char_id_sequence, logmel, delta, delta_delta))
					if len(char_id_sequence) > max_sentence_length:
						max_sentence_length = len(char_id_sequence)
					if logmel.shape[1] > max_logmel_length:
						max_logmel_length = logmel.shape[1]

	return dataset, max_sentence_length, max_logmel_length

def get_bucket_idx(length):
	return length // BUCKET_THRESHOLD

def get_duration_seconds(length):
	config = chainer.config
	return length * config.frame_shift

def load_buckets(buckets_limit, data_limit):
	if buckets_limit is not None:
		assert buckets_limit > 0
	if data_limit is not None:
		assert data_limit > 0

	wav_paths = [
		"/home/stark/sandbox/CSJ/WAV/core",
	]
	transcription_paths = [
		"/home/stark/sandbox/CSJ_/core",
	]
	data_cache_path = "/home/stark/sandbox/cache"

	mean_filename = os.path.join(data_cache_path, "mean.npy")
	std_filename = os.path.join(data_cache_path, "std.npy")	

	mean_x_batch = None
	stddev_x_batch = None

	# キャッシュが存在するか調べる
	cache_available = True
	for bucket_idx in xrange(buckets_limit):
		if os.path.isfile(os.path.join(data_cache_path, "feature_%d.npy" % bucket_idx)) == False:
			cache_available = False
			break
		if os.path.isfile(os.path.join(data_cache_path, "feature_length_%d.npy" % bucket_idx)) == False:
			cache_available = False
			break
		if os.path.isfile(os.path.join(data_cache_path, "sentence_%d.npy" % bucket_idx)) == False:
			cache_available = False
			break
		if os.path.isfile(os.path.join(data_cache_path, "sentence_length_%d.npy" % bucket_idx)) == False:
			cache_available = False
			break
	if os.path.isfile(mean_filename) == False:
		cache_available = False
	if os.path.isfile(std_filename) == False:
		cache_available = False

	if cache_available:
		print("loading dataset from cache ...")
		
		buckets_feature = []
		buckets_feature_length = []
		buckets_sentence = []
		buckets_sentence_length = []

		for bucket_idx in xrange(buckets_limit):
			feature_batch = np.load(os.path.join(data_cache_path, "feature_%d.npy" % bucket_idx))
			feature_length_batch = np.load(os.path.join(data_cache_path, "feature_length_%d.npy" % bucket_idx))
			sentence_batch = np.load(os.path.join(data_cache_path, "sentence_%d.npy" % bucket_idx))
			sentence_length_batch = np.load(os.path.join(data_cache_path, "sentence_length_%d.npy" % bucket_idx))

			buckets_feature.append(feature_batch)
			buckets_feature_length.append(feature_length_batch)
			buckets_sentence.append(sentence_batch)
			buckets_sentence_length.append(sentence_length_batch)

		mean_x_batch = np.load(mean_filename)
		stddev_x_batch = np.load(std_filename)

	else:
		dataset, max_sentence_length, max_logmel_length = load_audio_features_and_transcriptions(wav_paths, transcription_paths, buckets_limit, data_limit)

		if data_limit is not None:
			dataset = dataset[:data_limit]

		# 読み込んだデータをキャッシュ
		config = chainer.config
		try:
			os.mkdir(data_cache_path)
		except:
			pass

		# 必要なバケツの数を特定
		buckets_length = 0
		for idx, data in enumerate(dataset):
			sentence, logmel, delta, delta_delta = data
			assert logmel.shape[1] == delta.shape[1]
			assert delta.shape[1] == delta_delta.shape[1]
			audio_length = logmel.shape[1]
			bucket_idx = get_bucket_idx(audio_length)
			if bucket_idx > buckets_length:
				buckets_length = bucket_idx
		buckets_length += 1
		if buckets_limit is not None:
			buckets_length = buckets_limit if buckets_length > buckets_limit else buckets_length

		# バケツ中のデータの個数を特定
		buckets_volume = [0] * buckets_length

		# バケツ中のデータの最大長を特定
		valid_dataset_size = 0
		max_feature_length_for_bucket = [0] * buckets_length
		max_sentence_length_for_bucket = [0] * buckets_length
		for idx, data in enumerate(dataset):
			sentence, logmel, delta, delta_delta = data
			feature_length = logmel.shape[1]
			sentence_length = len(sentence)
			bucket_idx = get_bucket_idx(feature_length)
			if bucket_idx >= buckets_length:
				continue
			buckets_volume[bucket_idx] += 1
			valid_dataset_size += 1
			max_feature_length_for_bucket[bucket_idx] = BUCKET_THRESHOLD * (bucket_idx + 1) - 1
			assert feature_length <= max_feature_length_for_bucket[bucket_idx]
			if sentence_length > max_sentence_length_for_bucket[bucket_idx]:
				max_sentence_length_for_bucket[bucket_idx] = sentence_length

		# データの平均と標準偏差
		mean_x_batch = 0
		stddev_x_batch = 0

		# バケツにデータを格納
		buckets_feature = [None] * buckets_length
		buckets_feature_length = [None] * buckets_length
		buckets_sentence = [None] * buckets_length
		buckets_sentence_length = [None] * buckets_length

		for bucket_idx in xrange(buckets_length):
			num_data = buckets_volume[bucket_idx]
			max_feature_length = max_feature_length_for_bucket[bucket_idx]
			max_sentence_length = max_sentence_length_for_bucket[bucket_idx]
			buckets_feature[bucket_idx] = np.zeros((num_data, 3, config.num_mel_filters, max_feature_length), dtype=np.float32)
			buckets_feature_length[bucket_idx] = np.zeros((num_data,), dtype=np.int32)
			buckets_sentence[bucket_idx] = np.zeros((num_data, max_sentence_length), dtype=np.int32)
			buckets_sentence_length[bucket_idx] = np.zeros((num_data,), dtype=np.int32)

		for idx, data in enumerate(dataset):
			if idx % 100 == 0:
				sys.stdout.write("\r")
				sys.stdout.write(stdout.CLEAR)
				sys.stdout.write("creating buckets ({}/{}) ... ".format(idx + 1, len(dataset)))
				sys.stdout.flush()

			sentence, logmel, delta, delta_delta = data
			feature_length = logmel.shape[1]
			sentence_length = len(sentence)
			bucket_idx = get_bucket_idx(feature_length)
			if bucket_idx >= buckets_length:
				continue

			buckets_volume[bucket_idx] -= 1
			insert_idx = buckets_volume[bucket_idx]

			# 音響特徴量
			feature_batch = buckets_feature[bucket_idx]
			feature_batch[insert_idx, 0, :, :feature_length] = logmel			
			feature_batch[insert_idx, 1, :, :feature_length] = delta			
			feature_batch[insert_idx, 2, :, :feature_length] = delta_delta

			# 平均と標準偏差を計算
			mean_x_batch += np.mean(feature_batch[insert_idx, :, :, :feature_length], axis=2, keepdims=True) / valid_dataset_size
			stddev_x_batch += np.std(feature_batch[insert_idx, :, :, :feature_length], axis=2, keepdims=True) / valid_dataset_size	

			# 書き起こし
			sentence_batch = buckets_sentence[bucket_idx]
			sentence_batch[insert_idx, :sentence_length] = sentence

			# 音響特徴量の有効長
			feature_length_batch = buckets_feature_length[bucket_idx]
			feature_length_batch[insert_idx] = feature_length

			# 書き起こしの有効長
			sentence_length_batch = buckets_sentence_length[bucket_idx]
			sentence_length_batch[insert_idx] = sentence_length

		# for bucket_idx in xrange(buckets_length):
		# 	feature_batch = buckets_feature[bucket_idx]
		# 	sentence_batch = buckets_sentence[bucket_idx]
		# 	if feature_batch is None:
		# 		continue
		# 	if sentence_batch is None:
		# 		continue
		# 	print(bucket_idx, feature_batch.shape, sentence_batch.shape, get_duration_seconds(feature_batch.shape[3]))

		# ディスクにキャッシュ
		for bucket_idx in xrange(buckets_length):
			feature_batch = buckets_feature[bucket_idx]
			feature_length_batch = buckets_feature_length[bucket_idx]
			sentence_batch = buckets_sentence[bucket_idx]
			sentence_length_batch = buckets_sentence_length[bucket_idx]
			np.save(os.path.join(data_cache_path, "feature_%d.npy" % bucket_idx), feature_batch)
			np.save(os.path.join(data_cache_path, "feature_length_%d.npy" % bucket_idx), feature_length_batch)
			np.save(os.path.join(data_cache_path, "sentence_%d.npy" % bucket_idx), sentence_batch)
			np.save(os.path.join(data_cache_path, "sentence_length_%d.npy" % bucket_idx), sentence_length_batch)
		np.save(mean_filename, mean_x_batch)
		np.save(std_filename, stddev_x_batch)

	# reshape
	mean_x_batch = mean_x_batch[None, ...]
	stddev_x_batch = stddev_x_batch[None, ...]

	return buckets_feature, buckets_feature_length, buckets_sentence, buckets_sentence_length, mean_x_batch, stddev_x_batch