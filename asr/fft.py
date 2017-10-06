from __future__ import division
import numpy as np
from python_speech_features import sigproc
from scipy.fftpack import dct

def compute_deltas(logmel):
	# ΔとΔΔを計算
	delta = compute_delta(logmel, 1)
	# delta = (np.roll(logmel, -1, axis=0) - logmel) / 2
	delta_delta = compute_delta(delta, 1)
	# delta_delta = (np.roll(delta, -1, axis=0) - delta) / 2

	# 不要な部分を削除
	# ΔΔまで計算すると末尾の2つは正しくない値になる
	logmel = logmel[:-2]
	delta = delta[:-2]
	delta_delta = delta_delta[:-2]

	return logmel, delta, delta_delta

def augment_specgram(pspec, change_speech_rate=True, change_vocal_tract=True):
	new_pspec = None
	# 話速歪み
	if change_speech_rate == True:
		speed = max(min(np.random.normal(1, 0.15), 1.2), 0.8)
		orig_length = len(pspec)
		new_length = int(orig_length / speed)
		assert new_length > 0
		dim = pspec.shape[1]
		new_pspec = np.empty((new_length, dim), dtype=np.float64)

		for t in range(new_length):
			i = int(t * speed)
			new_pspec[t] = pspec[i]

		pspec = new_pspec

	# 声道長歪み
	if change_vocal_tract == True:
		new_pspec = np.empty((new_length, dim), dtype=np.float64) if new_pspec is None else pspec.copy()
		ratio = max(min(np.random.normal(1, 0.15), 1.2), 0.8)
		for d in range(dim):
			i = int(d * ratio)
			if i < dim:
				new_pspec[:, d] = pspec[:, i]
			else:
				new_pspec[:, d] = pspec[:, -1]
		pspec = new_pspec

	return pspec

def get_specgram(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfft=512, preemph=0.97, winfunc=lambda x:np.ones((x,))):
	signal = sigproc.preemphasis(signal, preemph)
	frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
	pspec = sigproc.powspec(frames, nfft)
	return pspec

def compute_logmel(pspec, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512,lowfreq=0, highfreq=None, preemph=0.97,
		  winfunc=lambda x:np.ones((x,)), fbank=None):
	highfreq = highfreq or samplerate/2
	if fbank is None:
		fbank = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
	feat = np.dot(pspec,fbank.T)
	feat = np.where(feat == 0, np.finfo(float).eps, feat)
	return np.log(feat)

def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
	highfreq= highfreq or samplerate/2
	assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
	lowmel = hz2mel(lowfreq)
	highmel = hz2mel(highfreq)
	melpoints = np.linspace(lowmel,highmel,nfilt+2)
	bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

	fbank = np.zeros([nfilt,nfft//2+1])
	for j in range(0,nfilt):
		for i in range(int(bin[j]), int(bin[j+1])):
			fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
		for i in range(int(bin[j+1]), int(bin[j+2])):
			fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])

	for j in range(0,nfilt):
		if np.count_nonzero(fbank[j]) == 0:
			fbank[j] = (fbank[j - 1] + fbank[j + 1]) / 2
			
	return fbank                 
	
def hz2mel(hz):
	return 2595 * np.log10(1+hz/700.)
	
def mel2hz(mel):
	return 700*(10**(mel/2595.0)-1)

def compute_delta(feat, N):
	if N < 1:
		raise ValueError('N must be an integer >= 1')
	NUMFRAMES = len(feat)
	denominator = 2 * sum([i**2 for i in range(1, N+1)])
	delta_feat = np.empty_like(feat)
	padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
	for t in range(NUMFRAMES):
		delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator
	return delta_feat
