# coding: utf8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, argparse, time, cupy, math, os, binascii, signal
import chainer
import numpy as np
import chainer.functions as F
from tqdm import tqdm
from chainer import cuda
from model import load_model, load_config
from args import args
from asr.error import compute_minibatch_error
from asr.dataset import Dataset, AugmentationOption
from asr.utils import printb, printr, printc, bold
from asr.vocab import get_unigram_ids, ID_BLANK
from asr.config import configure

def formatted_error(error_values):
	errors = []
	for error in error_values:
		errors.append("%.2f" % error)
	return errors
	
def main():
	assert args.dataset_path is not None
	assert args.working_directory is not None

	# ワーキングディレクトリ
	try:
		os.mkdir(args.working_directory)
	except:
		pass

	config_filename = os.path.join(args.working_directory, "model.json")
	model_filename = os.path.join(args.working_directory, "model.hdf5")

	# データの読み込み
	vocab_token_ids, vocab_id_tokens = get_unigram_ids()
	vocab_size = len(vocab_token_ids)

	# 設定
	model = load_model(model_filename, config_filename)

	# バケツごとのミニバッチサイズ
	# 自動的に調整される
	batchsizes_dev = [256] * 30

	dataset = Dataset(args.dataset_path, batchsizes_dev, batchsizes_dev, args.buckets_limit, token_ids=vocab_token_ids, id_blank=ID_BLANK, 
		buckets_cache_size=0, apply_cmn=args.apply_cmn)

	augmentation = AugmentationOption()
	if args.augmentation:
		augmentation.change_vocal_tract = True
		augmentation.change_speech_rate = True
		augmentation.add_noise = True

	# モデル
	model = load_model(model_filename, config_filename)
	assert model is not None

	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu(args.gpu_device)

	xp = model.xp
	using_gpu = False if xp is np else True

	# ログ
	dataset.dump()

	# バッチサイズの調整
	print("Searching for the best batch size ...")
	batch_dev = dataset.get_development_batch_iterator(batchsizes_dev, augmentation=augmentation, gpu=using_gpu)
	for _ in range(50):
		for x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_id, group_idx in batch_dev:
			try:
				with chainer.using_config("train", False):
					y_batch = model(x_batch)
			except Exception as e:
				if isinstance(e, cupy.cuda.runtime.CUDARuntimeError):
					batchsizes_dev[bucket_id] = max(batchsizes_dev[bucket_id] - 16, 4)
					print("new batchsize {} for bucket {}".format(batchsizes_dev[bucket_id], bucket_id + 1))

	# 学習
	printb("[Evaluation]")

	# ノイズ無しデータでバリデーション
	batch_dev = dataset.get_development_batch_iterator(batchsizes_dev, augmentation=augmentation, gpu=using_gpu)
	buckets_errors = [[] for i in range(dataset.get_num_buckets())]

	for x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_id, group_idx in batch_dev:

		try:
			with chainer.using_config("train", False):
				y_batch = model(x_batch, split_into_variables=False)
				y_batch = xp.argmax(y_batch.data, axis=2)
				error = compute_minibatch_error(y_batch, t_batch, ID_BLANK, vocab_token_ids, vocab_id_tokens)
				buckets_errors[bucket_id].append(error)

		except Exception as e:
			printr("")
			printc("{} (bucket {})".format(str(e), bucket_id + 1), color="red")
			
	# printr("")
	avg_errors_dev = []
	for errors in buckets_errors:
		avg_errors_dev.append(sum(errors) / len(errors) * 100)

	print("CER", formatted_error(avg_errors_dev))

if __name__ == "__main__":
	main()
