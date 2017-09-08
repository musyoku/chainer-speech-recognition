import chainer
import numpy as np

def configure():
	sampling_rate = 16000
	frame_width = 0.032
	config = chainer.global_config
	config.sampling_rate = sampling_rate
	config.frame_width = frame_width
	config.frame_shift = 0.01
	config.num_fft = int(sampling_rate * frame_width)
	config.num_mel_filters = 40
	config.window_func = lambda x:np.hanning(x)
	config.using_delta = True
	config.using_delta_delta = True
	config.bucket_split_sec = 0.5
	return config