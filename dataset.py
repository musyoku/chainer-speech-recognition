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
			logmel, delta, delta_delta = extract_features(signal, config.sampling_rate, config.num_fft, config.frame_width, config.frame_shift, config.num_mel_filter)
			dataset[idx] = signal, sentence, logmel, delta, delta_delta
		if logmel.shape[1] > max_width:
			max_width = logmel.shape[1]

	batch = np.zeros((batchsize, 3, config.num_mel_filter, max_width), dtype=np.float32)

	for batch_idx, data_idx in enumerate(indices):
		data = dataset[data_idx]
		signal, sentence, logmel, delta, delta_delta = data
		batch[batch_idx, 0, :, -logmel.shape[1]:] = logmel
		batch[batch_idx, 1, :, -delta.shape[1]:] = delta
		batch[batch_idx, 2, :, -delta_delta.shape[1]:] = delta_delta

	return batch

def extract_features(signal, sampling_rate=16000, nfft=512, winlen=0.032, winstep=0.01, nfilt=40, winfunc=lambda x:np.hanning(x)):
	# メルフィルタバンク出力の対数を計算
	logmel, energy = fbank(signal, sampling_rate, nfft=nfft, winlen=winlen, winstep=winstep, nfilt=nfilt, winfunc=winfunc)
	logmel = np.log(logmel)

	# ΔとΔΔを計算
	delta = (np.roll(logmel, -1, axis=0) - logmel) / 2
	delta_delta = (np.roll(delta, -1, axis=0) - delta) / 2

	# 不要な部分を削除
	# ΔΔまで計算すると末尾の2つは正しくない値になる
	logmel = logmel[:-2]
	delta = delta[:-2]
	delta_delta = delta_delta[:-2]

	return logmel.T, delta.T, delta_delta.T

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

					# ステレオの場合はchannelに従って選択
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

			accept_rate = 0.4	# ズレの割合がこれ以下なら教師データに誤りが含まれている可能性がある
			if trn_filename == "M03F0017.trn":
				accept_rate = 0.05
			for idx, (signal, sentence) in enumerate(batch):
				error = abs(len(signal) - num_points_per_character * len(sentence))
				rate = error / len(signal)
				if rate < accept_rate:
					raise Exception(len(signal), len(sentence), num_points_per_character, rate, trn_filename, idx + 1)

				dataset.append((signal, sentence))

	return dataset

def _load_audio_and_transcription():
	wav_paths = [
		"/media/stark/HDD/CSJ/WAV/core/",
		"/media/stark/HDD/CSJ/WAV/noncore/",
	]

	transcription_paths = [
		"/media/stark/HDD/CSJ_/core/",
		"/media/stark/HDD/CSJ_/noncore/",
	]

	min_sentence_length = 3

	debug_base_dir = None
	# debug_base_dir = "/home/aibo/wav/"
	debug_dump_figure = False

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
			except:
				print("{}をスキップしました（読み込みできません）".format(wav_filename))
				continue

			print("{}を読み込み中 ... shape={}, rate={}, min={}".format(wav_filename, audio.shape, sampling_rate, int(audio.size / sampling_rate / 60)))

			# 転記の読み込み
			sentence_batch = []
			channel_batch = []
			period_batch = []
			index_batch = []
			with codecs.open(os.path.join(trn_dir, trn_filename), "r", "utf-8") as f:
				data_index = -1
				for data in f:
					data_index += 1
					period_str, channel, sentence = data.split(":")
					if len(sentence) < min_sentence_length:
						continue
					period = period_str.split("-")
					start_sec, end_sec = float(period[0]), float(period[1])
					# print(start_sec, end_sec)
					sentence_batch.append(sentence.strip())
					period_batch.append((start_sec, end_sec))
					channel_batch.append(channel)
					index_batch.append(data_index)

			# 音声の切り出し
			signal_batch = []
			logmel_batch = []
			delta_batch = []
			delta_delta_batch = []
			for sentence, period, channel, idx in zip(sentence_batch, period_batch, channel_batch, index_batch):
				start_sec, end_sec = period
				start_frame = int(start_sec * sampling_rate)
				end_frame = int(end_sec * sampling_rate)
				assert start_frame <= len(audio)
				assert end_frame <= len(audio)
				signal = audio[start_frame:end_frame]
				assert len(signal) == end_frame - start_frame

				# ステレオの場合はchannelに従って選択
				if signal.ndim == 2:
					if channel == "S":	# 両方に含まれる場合
						signal = signal[:, 0]
					elif channel == "L":
						signal = signal[:, 0]
					elif channel == "R":
						signal = signal[:, 1]
					else:
						raise Exception()

				signal_batch.append(signal)

				# メルフィルタバンク出力の対数を計算
				feature, energy = fbank(signal, sampling_rate, nfft=512, winlen=0.032, winstep=0.01, nfilt=40, lowfreq=0, winfunc=lambda x:np.hanning(x))
				feature = np.log(feature)

				# ΔとΔΔを計算
				delta_feature = (np.roll(feature, -1, axis=0) - feature) / 2
				delta_delta_feature = (np.roll(delta_feature, -1, axis=0) - delta_feature) / 2

				# 不要な部分を削除
				# ΔΔまで計算すると末尾の2つは正しくない値になる
				feature = feature[:-2]
				delta_feature = delta_feature[:-2]
				delta_delta_feature = delta_delta_feature[:-2]

				logmel_batch.append(feature)
				delta_batch.append(delta_feature)
				delta_delta_batch.append(delta_delta_feature)

				# デバッグモードの場合は切り出した音声やスペクトログラムを保存する
				if debug_base_dir is not None:
					import pylab
					import scipy.signal
					from matplotlib import pyplot as plt

					debug_out_dir = debug_base_dir + wav_dir + re.sub(r"\..+", "", wav_filename)
					try:
						os.makedirs(debug_out_dir)
					except:
						pass

					if os.path.exists(os.path.join(debug_out_dir, "{}.wav").format(idx + 1)) == False:
						wavfile.write(os.path.join(debug_out_dir, "{}.wav".format(idx + 1)), sampling_rate, signal)
						with codecs.open(os.path.join(debug_out_dir, "{}.trn".format(idx + 1)), "w", "utf-8") as f:
							f.write(sentence)

						if debug_dump_figure:
							sampling_interval = 1.0 / sampling_rate
							times = np.arange(len(signal)) * sampling_interval
							pylab.clf()
							plt.rcParams['font.size'] = 20
							pylab.figure(figsize=(len(signal) / 2000, 16)) 

							ax1 = pylab.subplot(511)
							pylab.plot(times, signal)
							pylab.title("Waveform")
							pylab.xlabel("Time [sec]")
							pylab.ylabel("Amplitude")
							pylab.xlim([0, len(signal) * sampling_interval])

							ax2 = pylab.subplot(512)
							spectrum, freqs, bins, im = pylab.specgram(signal, 256, sampling_rate, noverlap=0.01 * sampling_rate, window=pylab.window_hanning, cmap=pylab.get_cmap("jet"))
							pylab.title("Spectrogram")
							pylab.xlabel("Time [sec]")
							pylab.ylabel("Frequency [Hz]")
							# pylab.colorbar()
							
							ax3 = pylab.subplot(513)
							pylab.pcolormesh(np.arange(0, feature.shape[0]), np.arange(1, 41), feature.T, cmap=pylab.get_cmap("jet"))
							pylab.title("Log mel filter bank features")
							pylab.xlabel("Frame")
							pylab.ylabel("Filter number")
							# pylab.colorbar()
							
							ax4 = pylab.subplot(514)
							pylab.pcolormesh(np.arange(0, delta_feature.shape[0]), np.arange(1, 41), delta_feature.T, cmap=pylab.get_cmap("jet"))
							pylab.title("Deltas")
							pylab.xlabel("Frame")
							pylab.ylabel("Filter number")
							# pylab.colorbar()
							
							ax5 = pylab.subplot(515)
							pylab.pcolormesh(np.arange(0, delta_delta_feature.shape[0]), np.arange(1, 41), delta_delta_feature.T, cmap=pylab.get_cmap("jet"))
							pylab.title("Delta-deltas")
							pylab.xlabel("Frame")
							pylab.ylabel("Filter number")
							# pylab.colorbar()

							pylab.tight_layout()
							pylab.savefig(os.path.join(debug_out_dir, "{}.png".format(idx + 1)), bbox_inches="tight")

			# デバッグ用ファイルの生成
			if debug_base_dir is not None:
				if os.path.exists(os.path.join(debug_out_dir, "0.trn")) == False:
					with codecs.open(os.path.join(debug_out_dir, "0.trn"), "w", "utf-8") as f:
						f.write("\n".join(sentence_batch))

			# 信号長と転記文字列長の不自然な部分を検出
			num_points_per_character = 0	# 1文字あたりの信号の数
			for signal, sentence in zip(signal_batch, sentence_batch):
				num_points_per_character += len(signal) / len(sentence)
			num_points_per_character /= len(signal)

			accept_rate = 0.4	# ズレの割合がこれ以下なら教師データに誤りが含まれている可能性がある
			if trn_filename == "M03F0017.trn":
				accept_rate = 0.05
			for signal, sentence, idx in zip(signal_batch, sentence_batch, index_batch):
				error = abs(len(signal) - num_points_per_character * len(sentence))
				rate = error / len(signal)
				if rate < accept_rate:
					raise Exception(len(signal), len(sentence), num_points_per_character, rate, trn_filename, idx + 1)

			for sentence, logmel, delta, delta_delta in zip(sentence_batch, logmel_batch, delta_batch, delta_delta_batch):
				dataset.append((sentence, logmel, delta, delta_delta))

	return dataset