# coding: utf-8
from __future__ import division
from __future__ import print_function
import os, codecs, re, sys
import chainer
import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import logfbank
from python_speech_features import fbank

def get_minibatch(bucket, dataset, batchsize):
	assert len(bucket) >= batchsize
	config = chainer.config
	indices = bucket[:batchsize]
	max_width = 0

	for idx in indices:
		data = dataset[idx]
		signal, sentence, logmel, delta, delta_delta = data
		if logmel is None:
			logmel, delta, delta_delta = extract_features(signal, config.sampling_rate, config.num_fft, config.frame_width, config.frame_shift, config.num_mel_filters, config.window_func, config.using_delta, config.using_delta_delta)
			dataset[idx] = signal, sentence, logmel, delta, delta_delta
		if logmel.shape[1] > max_width:
			max_width = logmel.shape[1]

	batch = np.zeros((batchsize, 3, config.num_mel_filters, max_width), dtype=np.float32)

	for batch_idx, data_idx in enumerate(indices):
		data = dataset[data_idx]
		signal, sentence, logmel, delta, delta_delta = data
		batch[batch_idx, 0, :, -logmel.shape[1]:] = logmel
		batch[batch_idx, 1, :, -delta.shape[1]:] = delta
		batch[batch_idx, 2, :, -delta_delta.shape[1]:] = delta_delta

	return batch

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

def load_audio_and_transcription(wav_paths, transcription_paths):
	assert len(wav_paths) > 0
	assert len(transcription_paths) > 0

	dataset = []

	for wav_dir, trn_dir in zip(wav_paths, transcription_paths):
		wav_fs = os.listdir(wav_dir)
		trn_fs = os.listdir(trn_dir)
		wav_fs.sort()
		trn_fs.sort()
		assert len(wav_fs) == len(trn_fs)

		for wav_filename, trn_filename in zip(wav_fs, trn_fs):
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

			print("loading {} ... shape={}, rate={}, min={}".format(wav_filename, audio.shape, sampling_rate, int(audio.size / sampling_rate / 60)))

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

					batch.append((signal, sentence))

			# 信号長と転記文字列長の不自然な部分を検出
			num_points_per_character = 0	# 1文字あたりの信号の数
			for signal, sentence in batch:
				num_points_per_character += len(signal) / len(sentence)
			num_points_per_character /= len(signal)

			accept_rate = 0.4	# ズレの割合がこれ以下なら教師データに誤りが含まれている可能性があるので目視で確認すべき
			if trn_filename == "M03F0017.trn":	# CSJのこのファイルだけ異常な早口がある
				accept_rate = 0.05
			for idx, (signal, sentence) in enumerate(batch):
				error = abs(len(signal) - num_points_per_character * len(sentence))
				rate = error / len(signal)
				if rate < accept_rate:
					raise Exception(len(signal), len(sentence), num_points_per_character, rate, trn_filename, idx + 1)

				dataset.append((signal, sentence))

	return dataset