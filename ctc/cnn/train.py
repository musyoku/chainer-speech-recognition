# coding: utf8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, argparse, time, cupy, math, os
import chainer
import numpy as np
import chainer.functions as F
from chainer import cuda, serializers
import asyncio
sys.path.append("../../")
import config
from error import compute_minibatch_error
from dataset import Dataset, cache_path, get_vocab, AugmentationOption, DevMinibatchIterator
from model import load_model, save_model, build_model
from util import stdout, print_bold
from optim import get_current_learning_rate, decay_learning_rate, get_optimizer

def formatted_error(error_values):
	errors = []
	for error in error_values:
		errors.append("%.2f" % error)
	return errors

def main():
	# データの読み込み
	vocab, vocab_inv, BLANK = get_vocab()
	vocab_size = len(vocab)

	# ミニバッチを取れないものは除外
	# GTX 1080 1台基準
	batchsizes = [32, 32, 32, 24, 16, 16, 12, 12, 8, 8, 8, 8, 8, 8, 8, 8]

	dataset = Dataset(cache_path, batchsizes, args.buckets_limit, id_blank=BLANK)
	dataset.dump_information()

	augmentation = AugmentationOption()
	if args.augmentation:
		augmentation.change_vocal_tract = True
		augmentation.change_speech_rate = True
		augmentation.add_noise = True


	total_iterations_train = dataset.get_total_training_iterations()

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
					x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx = dataset.get_minibatch(option=augmentation, gpu=True)

					# 誤差の計算
					y_batch = model(x_batch)	# list of variables
					loss = F.connectionist_temporal_classification(y_batch, t_batch, BLANK, x_length_batch, t_length_batch)

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
			# ノイズ無しデータでバリデーション
			iterator = DevMinibatchIterator(dataset, batchsizes, AugmentationOption(), gpu=args.gpu_device >= 0)
			buckets_errors = []
			for batch in iterator:
				try:
					x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx, group_idx = batch

					sys.stdout.write("\r" + stdout.CLEAR)
					sys.stdout.write("computing CER of bucket {} (group {})".format(bucket_idx + 1, group_idx + 1))
					sys.stdout.flush()

					y_batch = model(x_batch, split_into_variables=False)
					y_batch = xp.argmax(y_batch.data, axis=2)
					error = compute_minibatch_error(y_batch, t_batch, BLANK)

					while bucket_idx >= len(buckets_errors):
						buckets_errors.append([])

					buckets_errors[bucket_idx].append(error)
				except Exception as e:
					print(" ", bucket_idx, str(e))

			avg_errors_dev = []
			for errors in buckets_errors:
				avg_errors_dev.append(sum(errors) / len(errors) * 100)


		sys.stdout.write(stdout.MOVE)
		sys.stdout.write(stdout.LEFT)

		# ログ
		elapsed_time = time.time() - start_time
		total_time += elapsed_time
		print("Epoch {} done in {} min - total {} min".format(epoch, int(elapsed_time / 60), int(total_time / 60)))
		sys.stdout.write(stdout.CLEAR)
		print("	loss:", sum_loss / total_iterations_train)
		print("	CER:	{}".format(formatted_error(avg_errors_dev)))
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
	parser.add_argument("--augmentation", "-augmentation", default=False, action="store_true")
	
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
	parser.add_argument("--model-dir", "-m", type=str, default="model")

	parser.add_argument("--buckets-limit", type=int, default=None)
	parser.add_argument("--data-limit", type=int, default=None)
	parser.add_argument("--seed", "-seed", type=int, default=0)
	args = parser.parse_args()

	main()