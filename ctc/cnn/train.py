# coding: utf8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, argparse, time, cupy, math, os
import chainer
import numpy as np
import chainer.functions as F
from chainer import optimizers, cuda, serializers
sys.path.append("../../")
import config
from error import compute_minibatch_error
from dataset import Dataset, cache_path, get_vocab, AugmentationOption
from model import load_model, save_model, build_model
from ctc import connectionist_temporal_classification

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"
	MOVE = "\033[1A"
	LEFT = "\033[G"

def print_bold(str):
	print(stdout.BOLD + str + stdout.END)

def get_current_learning_rate(opt):
	if isinstance(opt, optimizers.NesterovAG):
		return opt.lr
	if isinstance(opt, optimizers.MomentumSGD):
		return opt.lr
	if isinstance(opt, optimizers.SGD):
		return opt.lr
	if isinstance(opt, optimizers.Adam):
		return opt.alpha
	raise NotImplementedError()

def get_optimizer(name, lr, momentum):
	if name == "sgd":
		return optimizers.SGD(lr=lr)
	if name == "msgd":
		return optimizers.MomentumSGD(lr=lr, momentum=momentum)
	if name == "nesterov":
		return optimizers.NesterovAG(lr=lr, momentum=momentum)
	if name == "adam":
		return optimizers.Adam(alpha=lr, beta1=momentum)
	raise NotImplementedError()

def decay_learning_rate(opt, factor, final_value):
	if isinstance(opt, optimizers.NesterovAG):
		if opt.lr <= final_value:
			return final_value
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.SGD):
		if opt.lr <= final_value:
			return final_value
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.MomentumSGD):
		if opt.lr <= final_value:
			return final_value
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.Adam):
		if opt.alpha <= final_value:
			return final_value
		opt.alpha *= factor
		return
	raise NotImplementedError()

def formatted_error(error_values):
	errors = []
	for error in error_values:
		errors.append("%.2f" % error)
	return errors

def main():
	# データの読み込み
	vocab, vocab_inv, BLANK = get_vocab()
	vocab_size = len(vocab)

	dataset = Dataset(cache_path, args.buckets_limit, id_blank=BLANK)
	dataset.dump_information()
	augmentation = AugmentationOption()
	augmentation.change_vocal_tract = False
	augmentation.change_speech_rate = False
	augmentation.add_noise = False

	# ミニバッチを取れないものは除外
	# GTX 1080 1台基準
	batchsizes = [32, 32, 32, 24, 16, 16, 12, 12, 8, 8, 8, 8, 8, 8, 8, 8]

	total_iterations_train = dataset.get_total_training_iterations(batchsizes)

	# モデル
	chainer.global_config.vocab_size = vocab_size
	chainer.global_config.ndim_audio_features = args.ndim_audio_features
	chainer.global_config.ndim_h = args.ndim_h
	chainer.global_config.ndim_dense = args.ndim_dense
	chainer.global_config.num_conv_layers = args.num_conv_layers
	chainer.global_config.kernel_size = (3, 5)
	chainer.global_config.dropout = args.dropout
	chainer.global_config.weightnorm = args.weightnorm
	chainer.global_config.wgain = args.wgain
	chainer.global_config.architecture = args.architecture

	model = load_model(args.model_dir)
	if model is None:
		config = chainer.config
		model = build_model(vocab_size=vocab_size, ndim_audio_features=config.ndim_audio_features, 
		 ndim_h=config.ndim_h, ndim_dense=config.ndim_dense, num_conv_layers=config.num_conv_layers,
		 kernel_size=(3, 5), dropout=config.dropout, weightnorm=config.weightnorm, wgain=config.wgain,
		 num_mel_filters=config.num_mel_filters, architecture=config.architecture)

	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu(args.gpu_device)
	xp = model.xp

	# optimizer
	optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
	final_learning_rate = 1e-4
	total_time = 0

	for epoch in xrange(1, args.total_epoch + 1):
		print_bold("Epoch %d" % epoch)
		start_time = time.time()
		loss_value = 0
		sum_loss = 0
		
		with chainer.using_config("train", True):
			for itr in xrange(1, total_iterations_train + 1):
				try:
					x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx = dataset.get_minibatch(batchsizes, option=augmentation, gpu=True)

					# 誤差の計算
					y_batch = model(x_batch)	# list of variables
					loss = connectionist_temporal_classification(y_batch, t_batch, BLANK, x_length_batch, t_length_batch)

					# NaN
					loss_value = float(loss.data)
					if loss_value != loss_value:
						raise Exception("Encountered NaN when computing loss.")

					# 更新
					optimizer.update(lossfun=lambda: loss)

				except Exception as e:
					print(" ", bucket_idx, str(e))

				sum_loss += loss_value
				sys.stdout.write("\r" + stdout.CLEAR)
				sys.stdout.write("\riteration {}/{}".format(itr, total_iterations_train))
				sys.stdout.flush()

		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()
		save_model(args.model_dir, model)

		# バリデーション
		with chainer.using_config("train", False):
			x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx = dataset.get_minibatch(batchsizes, option=augmentation, gpu=True)
			y_batch = model(x_batch, split_into_variables=False)
			y_batch = xp.argmax(y_batch.data, axis=2)
			train_error = compute_minibatch_error(y_batch, t_batch, BLANK) * 100

		sys.stdout.write(stdout.MOVE)
		sys.stdout.write(stdout.LEFT)

		# ログ
		elapsed_time = time.time() - start_time
		total_time += elapsed_time
		print("Epoch {} done in {} min - total {} min".format(epoch, int(elapsed_time / 60), int(total_time / 60)))
		sys.stdout.write(stdout.CLEAR)
		print("	loss:", sum_loss / total_iterations_train)
		print("	CER (train):	{:.2f}".format(train_error))
		print("	lr: {}".format(get_current_learning_rate(optimizer)))

		# 学習率の減衰
		decay_learning_rate(optimizer, args.lr_decay, final_learning_rate)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--total-epoch", "-e", type=int, default=1000)
	parser.add_argument("--grad-clip", "-gc", type=float, default=1) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=1e-5) 
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.001)
	parser.add_argument("--lr-decay", "-decay", type=float, default=0.95)
	parser.add_argument("--momentum", "-mo", type=float, default=0.9)
	parser.add_argument("--optimizer", "-opt", type=str, default="adam")
	
	parser.add_argument("--ndim-audio-features", "-features", type=int, default=3)
	parser.add_argument("--ndim-h", "-dh", type=int, default=320)
	parser.add_argument("--ndim-dense", "-dd", type=int, default=1024)
	parser.add_argument("--num-conv-layers", "-nconv", type=int, default=4)
	parser.add_argument("--num-dense-layers", "-ndense", type=int, default=4)
	parser.add_argument("--wgain", "-w", type=float, default=1)

	parser.add_argument("--nonlinear", type=str, default="relu")
	parser.add_argument("--dropout", "-dropout", type=float, default=0)
	parser.add_argument("--weightnorm", "-weightnorm", default=False, action="store_true")
	parser.add_argument("--architecture", "-arch", type=str, default="zhang")
	
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--interval", type=int, default=100)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--dev-split", "-split", type=float, default=0.01)
	parser.add_argument("--train-filename", "-train", default=None)
	parser.add_argument("--dev-filename", "-dev", default=None)

	parser.add_argument("--buckets-limit", type=int, default=None)
	parser.add_argument("--data-limit", type=int, default=None)
	parser.add_argument("--seed", "-seed", type=int, default=0)
	args = parser.parse_args()

	main()
