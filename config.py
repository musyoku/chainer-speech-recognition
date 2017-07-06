import chainer
import numpy as np

sampling_rate = 16000
frame_width = 0.032		# sec
frame_shift = 0.01		# sec
gpu_ids = [0, 1, 3]		# 複数GPUを使う場合
num_fft = int(sampling_rate * frame_width)
num_mel_filters = 40

# audio
chainer.global_config.sampling_rate = sampling_rate
chainer.global_config.frame_width = frame_width
chainer.global_config.frame_shift = frame_shift
chainer.global_config.num_fft = num_fft
chainer.global_config.num_mel_filters = num_mel_filters
chainer.global_config.window_func = lambda x:np.hanning(x)
chainer.global_config.using_delta = True
chainer.global_config.using_delta_delta = True

# training
chainer.global_config.bucket_split_sec = 0.5