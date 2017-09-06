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
from dataset import Dataset, cache_path, AugmentationOption
from model import load_model
from util import printb, printr
from vocab import load_unigram_and_bigram_ids, ID_BLANK

def main():
	vocab_token_ids, vocab_id_tokens = load_unigram_and_bigram_ids("../../bigram.list")

	# ミニバッチを取れないものは除外
	# GTX 1080 1台基準
	batchsizes = [32, 32, 32, 24, 16, 16, 12, 12, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

	dataset = Dataset(cache_path, batchsizes, args.buckets_limit, token_ids=vocab_token_ids, id_blank=ID_BLANK)
	dataset.dump_information()

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
	statistics = xp.zeros((len(vocab_token_ids),), dtype=xp.int32)

	with chainer.using_config("train", False):
		iterator = dataset.get_iterator_dev(batchsizes, None, gpu=args.gpu_device >= 0)
		buckets_errors = []
		for batch in iterator:
			try:
				x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx, group_idx = batch

				printr("computing CER of bucket {} (group {})".format(bucket_idx + 1, group_idx + 1))

				y_batch = model(x_batch, split_into_variables=False)
				y_batch = xp.argmax(y_batch.data, axis=2)
				y_batch = xp.ravel(y_batch)
				nonzero = xp.nonzero(y_batch)
				for index in nonzero:
					token_id = y_batch[index]
					statistics[token_id] += 1

			except Exception as e:
				print(" ", bucket_idx, str(e))

		printr("")
		ranking = {}
		for token_id in range(len(vocab_id_tokens)):
			ranking[vocab_id_tokens[token_id]] = statistics[token_id]
		for token, count in sorted(ranking.items(), key=lambda x:x[1]):
			print(token, count)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--buckets-limit", type=int, default=None)
	parser.add_argument("--seed", "-seed", type=int, default=0)
	parser.add_argument("--augmentation", "-augmentation", default=False, action="store_true")
	args = parser.parse_args()
	main()
