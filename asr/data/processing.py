import chainer, codecs, jaconv
import numpy as np
from chainer import cuda
from .. import fft
from ..vocab import convert_sentence_to_unigram_tokens
from ..utils import printb, printr, printc

def generate_signal_transcription_pairs(trn_path, audio, sampling_rate, num_fft):
	batch = []
	with codecs.open(trn_path, "r", "utf-8") as f:
		for data in f:
			period_str, channel, sentence = data.split(":")
			sentence = sentence.strip()
			period = period_str.split("-")
			start_sec, end_sec = float(period[0]), float(period[1])
			start_frame = int(start_sec * sampling_rate)
			end_frame = int(end_sec * sampling_rate)

			assert start_frame <= len(audio)
			assert end_frame <= len(audio)

			signal = audio[start_frame:end_frame]

			assert len(signal) == end_frame - start_frame
			if len(signal) < num_fft * 3:
				printr("")
				print("{} skipped. (length={})".format(sentence, len(signal)))
				continue

			# channelに従って選択
			if signal.ndim == 2:
				if channel == "S":	# 両方に含まれる場合はL
					signal = signal[:, 0]
				elif channel == "L":
					signal = signal[:, 0]
				elif channel == "R":
					signal = signal[:, 1]
				else:
					raise Exception()

			batch.append((signal, sentence))
	return batch
	
class Processor():

	def __init__(self, sampling_rate=16000, frame_width=0.032, frame_shift=0.01, num_mel_filters=40, window_func="hanning",
		using_delta=True, using_delta_delta=True):
		assert window_func in ["hanning", "hamming"]
		self.sampling_rate = sampling_rate
		self.frame_width = frame_width
		self.sampling_rate = sampling_rate
		self.frame_width = frame_width
		self.frame_shift = frame_shift
		self.num_fft = int(sampling_rate * frame_width)
		self.num_mel_filters = num_mel_filters

		if window_func == "hanning":
			self.window_func = lambda x:np.hanning(x)
		elif winfunc == "hamming":
			self.window_func = lambda x:np.hamming(x)

		self.using_delta = using_delta
		self.using_delta_delta = using_delta_delta

		self.fbank = fft.get_filterbanks(nfft=self.num_fft, nfilt=num_mel_filters, samplerate=sampling_rate)

	def extract_batch_features(self, batch, augmentation=None, apply_cmn=False):
		max_feature_length = 0
		max_sentence_length = 0
		audio_features = []
		sentences = []

		for signal, sentence in batch:
			# データ拡大
			if augmentation and augmentation.add_noise:
				gain = max(min(np.random.normal(200, 100), 500), 0)
				noise = acoustics.generator.noise(len(signal), color="white") * gain
				signal += noise.astype(np.int16)

			specgram = fft.get_specgram(signal, self.sampling_rate, nfft=self.num_fft, winlen=self.frame_width, winstep=self.frame_shift, winfunc=self.window_func)
			
			# データ拡大
			if augmentation and augmentation.using_augmentation():
				specgram = fft.augment_specgram(specgram, augmentation.change_speech_rate, augmentation.change_vocal_tract)

			# CMN 
			if apply_cmn:
				log_specgram = np.log(specgram)
				specgram = np.exp(np.log(specgram) - np.mean(log_specgram, axis=0))

			logmel = fft.compute_logmel(specgram, self.sampling_rate, fbank=self.fbank, nfft=self.num_fft, winlen=self.frame_width, winstep=self.frame_shift, nfilt=self.num_mel_filters, winfunc=self.window_func)
			logmel, delta, delta_delta = fft.compute_deltas(logmel)

			logmel = logmel.T
			delta = delta.T
			delta_delta = delta_delta.T

			if logmel.shape[1] > max_feature_length:
				max_feature_length = logmel.shape[1]
			if len(sentence) > max_sentence_length:
				max_sentence_length = len(sentence)

			if logmel.shape[1] == 0:
				continue

			audio_features.append((logmel, delta, delta_delta))
			sentence = jaconv.hira2kata(sentence) # 強制カタカナ変換
			sentences.append(sentence)

		assert max_feature_length > 0
		return audio_features, sentences, max_feature_length, max_sentence_length

	def features_to_minibatch(self, features, sentences, max_feature_length, max_sentence_length, token_ids, id_blank):
		assert isinstance(token_ids, dict)
		assert isinstance(id_blank, int)
		batchsize = len(features)
		channels = 1
		if self.using_delta:
			channels += 1
		if self.using_delta_delta:
			channels += 1
		height = self.num_mel_filters

		x_batch = np.zeros((batchsize, channels, height, max_feature_length), dtype=np.float32)
		t_batch = np.full((batchsize, max_sentence_length), id_blank, dtype=np.int32)
		bigram_batch = np.full((batchsize, max_sentence_length), id_blank, dtype=np.int32)
		x_length_batch = []
		t_length_batch = []

		for batch_idx, ((logmel, delta, delta_delta), sentence) in enumerate(zip(features, sentences)):
			# 書き起こしをunigram単位に分割
			unigram_tokens = convert_sentence_to_unigram_tokens(sentence)
			bigram_tokens = []
			if len(unigram_tokens) > 1:
				for first, second in zip(unigram_tokens[:-1], unigram_tokens[1:]):
					bigram_tokens.append(first + second)

			unigram_ids = []
			bigram_ids = [-1]
			for token in unigram_tokens:
				unigram_ids.append(token_ids[token])
			for token in bigram_tokens:
				if token in token_ids:
					bigram_ids.append(token_ids[token])
				else:
					bigram_ids.append(-1)
			assert len(unigram_ids) == len(bigram_ids)

			x_length = logmel.shape[1]
			t_length = len(unigram_ids)

			x_batch[batch_idx, 0, :, :x_length] = logmel
			if self.using_delta:
				x_batch[batch_idx, 1, :, :x_length] = delta
			if self.using_delta_delta:
				x_batch[batch_idx, 2, :, :x_length] = delta_delta
			x_length_batch.append(x_length)

			# CTCが適用可能かチェック
			num_trans_same_label = np.count_nonzero(unigram_ids == np.roll(unigram_ids, 1))
			required_length = t_length * 2 + 1 + num_trans_same_label
			if x_length < required_length:
				possibole_t_length = (x_length - num_trans_same_label - 1) // 2
				unigram_ids = unigram_ids[:possibole_t_length]
				bigram_ids = bigram_ids[:possibole_t_length]
				t_length = len(unigram_ids)

			# t
			t_batch[batch_idx, :t_length] = unigram_ids
			bigram_batch[batch_idx, :t_length] = bigram_ids
			t_length_batch.append(t_length)

		return x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch