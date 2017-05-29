# coding: utf-8
import pylab, argparse, codecs, os
import scipy.signal
import scipy.io.wavfile as wavfile
import numpy as np
from matplotlib import pyplot as plt
from python_speech_features import logfbank
from python_speech_features import fbank

def plot_features(out_dir, signal, logmel, delta, delta_delta, sampling_rate, filename):
	try:
		os.makedirs(out_dir)
	except:
		pass

	sampling_interval = 1.0 / sampling_rate
	times = np.arange(len(signal)) * sampling_interval
	pylab.clf()
	plt.rcParams['font.size'] = 18
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
	pylab.pcolormesh(np.arange(0, logmel.shape[0]), np.arange(1, 41), logmel.T, cmap=pylab.get_cmap("jet"))
	pylab.title("Log mel filter bank features")
	pylab.xlabel("Frame")
	pylab.ylabel("Filter number")
	# pylab.colorbar()
	
	ax4 = pylab.subplot(514)
	pylab.pcolormesh(np.arange(0, delta.shape[0]), np.arange(1, 41), delta.T, cmap=pylab.get_cmap("jet"))
	pylab.title("Deltas")
	pylab.xlabel("Frame")
	pylab.ylabel("Filter number")
	# pylab.colorbar()
	
	ax5 = pylab.subplot(515)
	pylab.pcolormesh(np.arange(0, delta_delta.shape[0]), np.arange(1, 41), delta_delta.T, cmap=pylab.get_cmap("jet"))
	pylab.title("Delta-deltas")
	pylab.xlabel("Frame")
	pylab.ylabel("Filter number")
	# pylab.colorbar()

	pylab.tight_layout()
	pylab.savefig(os.path.join(out_dir, filename), bbox_inches="tight")

def split_audio(wav_filename, trn_filename):
	sampling_rate, audio = wavfile.read(wav_filename)
	dataset = []
	with codecs.open(trn_filename, "r", "utf-8") as f:
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

			dataset.append((signal, sentence, start_sec, end_sec, channel))

	features = []
	for signal, sentence, start_sec, end_sec, channel in dataset:
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

		# メルフィルタバンク出力の対数を計算
		logmel, energy = fbank(signal, sampling_rate, nfft=512, winlen=0.032, winstep=0.01, nfilt=40, lowfreq=0, winfunc=lambda x:np.hanning(x))
		logmel = np.log(logmel)

		# ΔとΔΔを計算
		delta = (np.roll(logmel, -1, axis=0) - logmel) / 2
		delta_delta = (np.roll(delta, -1, axis=0) - delta) / 2

		# 不要な部分を削除
		# ΔΔまで計算すると末尾の2つは正しくない値になる
		logmel = logmel[:-2]
		delta = delta[:-2]
		delta_delta = delta_delta[:-2]

		features.append((signal, logmel, delta, delta_delta))

	return sampling_rate, features

def main(args):
	sampling_rate, features = split_audio(args.wav_filename, args.transcription_filename)
	for idx, feature in enumerate(features):
		signal, logmel, delta, delta_delta = feature
		plot_features(args.out_dir, signal, logmel, delta, delta_delta, sampling_rate, "{}.png".format(idx+1))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--wav-filename", "-wav", type=str)
	parser.add_argument("--transcription-filename", "-trn", type=str)
	parser.add_argument("--out-dir", "-out", type=str)
	args = parser.parse_args()
	main(args)