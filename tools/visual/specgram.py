import pylab, argparse, codecs, os, sys, pickle
import scipy.signal
import scipy.io.wavfile as wavfile
import os
import numpy as np
from matplotlib import pyplot as plt
from acoustics import generator
sys.path.append(os.path.join("..", ".."))
from asr import fft

def plot_features(out_dir, signal, sampling_rate, filename, apply_cmn=False, global_normalization=False, add_noise=False):
	try:
		os.makedirs(out_dir)
	except:
		pass

	# add noise (optional)
	if add_noise:
		noise = (generator.noise(len(signal), color="white") * 400).astype(np.int16)
		signal += noise	

	# .wav
	# wavfile.write(os.path.join(out_dir, filename + ".wav"), sampling_rate, signal.astype(np.int16))

	specgram = fft.get_specgram(signal, sampling_rate, nfft=512, winlen=args.frame_width, winstep=args.frame_shift, winfunc=lambda x:np.hanning(x))
	log_specgram = np.log(specgram)

	if apply_cmn:
		specgram = np.exp(np.log(specgram) - np.mean(log_specgram, axis=0))

	# data augmentation
	# specgram = fft.augment_specgram(specgram)

	logmel = fft.compute_logmel(specgram, sampling_rate, nfft=512, winlen=args.frame_width, winstep=args.frame_shift, nfilt=args.num_mel_filters, lowfreq=0, winfunc=lambda x:np.hanning(x))
	logmel, delta, delta_delta = fft.compute_deltas(logmel)

	if global_normalization:
		mean_filename = os.path.join(args.stats_dir, "mean.npy")
		nvar_filename = os.path.join(args.stats_dir, "nvar.npy")
		total_filename = os.path.join(args.stats_dir, "total.count")

		mean = np.load(mean_filename).astype(np.float32)
		nvar = np.load(nvar_filename).astype(np.float32)
		with open(total_filename, mode="rb") as f:
			stats_total = pickle.load(f)

		mean = mean
		std = np.sqrt(nvar / (stats_total - 1))

		logmel = (logmel - mean[0]) / std[0]
		delta = (delta - mean[1]) / std[1]
		delta_delta = (delta_delta - mean[2]) / std[2]
		

	_plot_features(out_dir, signal, sampling_rate, logmel, delta, delta_delta, specgram, filename)

def _plot_features(out_dir, signal, sampling_rate, logmel, delta, delta_delta, specgram, filename):
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
	specgram = np.log(specgram)
	pylab.pcolormesh(np.arange(0, specgram.shape[0]), np.arange(0, specgram.shape[1]) * 8000 / specgram.shape[1], specgram.T, cmap=pylab.get_cmap("jet"))
	pylab.title("Spectrogram")
	pylab.xlabel("Time [sec]")
	pylab.ylabel("Frequency [Hz]")
	pylab.colorbar()
	
	ax3 = pylab.subplot(513)
	pylab.pcolormesh(np.arange(0, logmel.shape[0]), np.arange(1, args.num_mel_filters + 1), logmel.T, cmap=pylab.get_cmap("jet"))
	pylab.title("Log mel filter bank features")
	pylab.xlabel("Frame")
	pylab.ylabel("Filter number")
	pylab.colorbar()
	
	ax4 = pylab.subplot(514)
	pylab.pcolormesh(np.arange(0, delta.shape[0]), np.arange(1, args.num_mel_filters + 1), delta.T, cmap=pylab.get_cmap("jet"))
	pylab.title("Deltas")
	pylab.xlabel("Frame")
	pylab.ylabel("Filter number")
	pylab.colorbar()
	
	ax5 = pylab.subplot(515)
	pylab.pcolormesh(np.arange(0, delta_delta.shape[0]), np.arange(1, args.num_mel_filters + 1), delta_delta.T, cmap=pylab.get_cmap("jet"))
	pylab.title("Delta-deltas")
	pylab.xlabel("Frame")
	pylab.ylabel("Filter number")
	pylab.colorbar()

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

		features.append(signal)

	return sampling_rate, features

def main():
	assert args.wav_filename
	assert args.trn_filename
	assert args.out_dir
	# assert args.stats_dir

	sampling_rate, features = split_audio(args.wav_filename, args.trn_filename)
	for index, signal in enumerate(features):
		plot_features(args.out_dir, signal, sampling_rate, "{}.png".format(index + 1), apply_cmn=False)
		plot_features(args.out_dir, signal, sampling_rate, "{}.norm.png".format(index + 1), apply_cmn=False, 
			global_normalization=True)
		plot_features(args.out_dir, signal, sampling_rate, "{}.noise.png".format(index + 1), apply_cmn=False, 
			global_normalization=True, add_noise=True)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--wav-filename", "-wav", type=str)
	parser.add_argument("--trn-filename", "-trn", type=str)
	parser.add_argument("--out-dir", "-out", type=str)
	parser.add_argument("--stats-dir", "-stats", type=str)
	parser.add_argument("--frame-width", "-fwidth", type=float, default=0.032)
	parser.add_argument("--frame-shift", "-fshift", type=float, default=0.01)
	parser.add_argument("--num-mel-filters", "-nmel", type=int, default=40) 
	args = parser.parse_args()
	main()