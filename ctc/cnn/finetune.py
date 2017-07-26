# coding: utf8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, argparse, time, cupy, math, os, binascii
import chainer
import numpy as np
import chainer.functions as F
from chainer import cuda, serializers
from multiprocessing import Process, Queue
sys.path.append("../../")
import config
from error import compute_minibatch_error
from dataset import cache_path, get_vocab, AugmentationOption, TestMinibatchIterator
from model import load_model, save_model, build_model
from util import stdout, print_bold
from optim import get_current_learning_rate, decay_learning_rate, get_optimizer

def formatted_error(error_values):
	errors = []
	for error in error_values:
		errors.append("%.2f" % error)
	return errors

def preloading_loop(dataset, augmentation, num_load, queue):
	np.random.seed(int(binascii.hexlify(os.urandom(4)), 16))
	for i in range(num_load):
		queue.put(dataset.get_minibatch(option=augmentation, gpu=False))
	return queue

def main():
	# データの読み込み
	vocab, vocab_inv, BLANK = get_vocab()
	vocab_size = len(vocab)

	# ミニバッチを取れないものは除外
	# GTX 1080 1台基準
	batchsizes = [32, 32, 32, 24, 16, 16, 12, 12, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

	augmentation = AugmentationOption()
	if args.augmentation:
		augmentation.change_vocal_tract = True
		augmentation.change_speech_rate = True
		augmentation.add_noise = True
	
	model = load_model(args.model_dir)
	assert model is not None

	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu(args.gpu_device)
	xp = model.xp

	# optimizer
	optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

	wav_path = "/home/stark/sandbox/tuning/wav"
	trn_path = "/home/stark/sandbox/tuning/trn"

	with chainer.using_config("train", True):
		iterator = TestMinibatchIterator(wav_path, trn_path, cache_path, batchsizes, BLANK, option=augmentation, gpu=args.gpu_device >= 0)

		for epoch in xrange(1, args.total_epoch + 1):
			print_bold("Epoch %d" % epoch)
			start_time = time.time()
			loss_value = 0
			sum_loss = 0
			total_iteration = 0
			iterator.reset()
				
			buckets_errors = []
			for batch in iterator:
				try:
					x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx, progress = batch

					# 誤差の計算
					y_batch = model(x_batch)	# list of variables
					loss = F.connectionist_temporal_classification(y_batch, t_batch, BLANK, x_length_batch, t_length_batch)

					# NaN
					loss_value = float(loss.data)
					if loss_value != loss_value:
						raise Exception("Encountered NaN when computing loss.")

					# 更新
					optimizer.update(lossfun=lambda: loss)

					total_iteration += 1

				except Exception as e:
					print(" ", bucket_idx, str(e))

				sum_loss += loss_value
				sys.stdout.write("\r" + stdout.CLEAR)
				sys.stdout.write("\riteration {}".format(total_iteration))
				sys.stdout.flush()

			sys.stdout.write("\r" + stdout.CLEAR)
			sys.stdout.flush()
			save_model(args.model_dir, model)

			print("	loss:", sum_loss / total_iteration)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--total-epoch", "-e", type=int, default=1000)
	parser.add_argument("--grad-clip", "-gc", type=float, default=1) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=1e-5) 
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.00001)
	parser.add_argument("--lr-decay", "-decay", type=float, default=0.95)
	parser.add_argument("--momentum", "-mo", type=float, default=0.9)
	parser.add_argument("--optimizer", "-opt", type=str, default="sgd")

	parser.add_argument("--augmentation", "-augmentation", default=False, action="store_true")
	
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	args = parser.parse_args()

	main()
