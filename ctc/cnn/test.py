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
from dataset import wav_path_test, trn_path_test, cache_path, get_vocab, AugmentationOption, TestMinibatchIterator
from model import load_model
from util import stdout, print_bold

def main():
	# データの読み込み
	vocab, vocab_inv, BLANK = get_vocab()
	vocab_size = len(vocab)

	# ミニバッチを取れないものは除外
	# GTX 1080 1台基準
	batchsizes = [96, 64, 64, 64, 64, 64, 64, 64, 48, 48, 48, 32, 32, 24, 24, 24]

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

	# テスト
	with chainer.using_config("train", False):
		iterator = TestMinibatchIterator(wav_path_test, trn_path_test, cache_path, batchsizes, BLANK, buckets_limit=args.buckets_limit, option=augmentation, gpu=args.gpu_device >= 0)
		buckets_errors = []
		for batch in iterator:
			x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx, progress = batch

			sys.stdout.write("\r" + stdout.CLEAR)
			sys.stdout.write("computing CER of bucket {} ({} %)".format(bucket_idx + 1, int(progress * 100)))
			sys.stdout.flush()

			y_batch = model(x_batch, split_into_variables=False)
			y_batch = xp.argmax(y_batch.data, axis=2)
			error = compute_minibatch_error(y_batch, t_batch, BLANK, print_sequences=False, vocab=vocab_inv)

			while bucket_idx >= len(buckets_errors):
				buckets_errors.append([])

			buckets_errors[bucket_idx].append(error)

		avg_errors = []
		for errors in buckets_errors:
			avg_errors.append(sum(errors) / len(errors))

		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()

		print_bold("bucket	CER")
		for bucket_idx, error in enumerate(avg_errors):
			print("{}	{}".format(bucket_idx + 1, error * 100))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--buckets-limit", type=int, default=None)
	parser.add_argument("--seed", "-seed", type=int, default=0)
	parser.add_argument("--augmentation", "-augmentation", default=False, action="store_true")
	args = parser.parse_args()
	main()
