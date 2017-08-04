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
from dataset import Dataset, cache_path, AugmentationOption
from model import load_model, save_model, build_model, save_params
from util import stdout, print_bold
from optim import get_current_learning_rate, decay_learning_rate, get_optimizer
from vocab import load_all_tokens

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
	TOKENS, _, BLANK = load_all_tokens("../../triphone.list")
	vocab_size = len(TOKENS)

	# ミニバッチを取れないものは除外
	# GTX 1080 1台基準
	batchsizes = [32, 32, 32, 24, 16, 16, 12, 12, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

	cache_size = 0 if args.multiprocessing else 200	# マルチプロセスの場合キャッシュごとコピーされるのでメモリを圧迫する
	dataset = Dataset(cache_path, batchsizes, args.buckets_limit, id_blank=BLANK, num_buckets_to_store_memory=cache_size)
	dataset.dump_information()

	augmentation = AugmentationOption()
	if args.augmentation:
		augmentation.change_vocal_tract = True
		augmentation.change_speech_rate = True
		augmentation.add_noise = True

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
	save_params(args.model_dir)

	model = load_model(args.model_dir)
	if model is None:
		config = chainer.config
		model = build_model(vocab_size=vocab_size, ndim_audio_features=config.ndim_audio_features, 
		 ndim_h=config.ndim_h, ndim_dense=config.ndim_dense, num_conv_layers=config.num_conv_layers,
		 kernel_size=(3, 5), dropout=config.dropout, weightnorm=config.weightnorm, wgain=config.wgain,
		 num_mel_filters=config.num_mel_filters, architecture=config.architecture)

	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu(args.gpu_device)
	xp = model.xp

	# optimizer
	optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
	final_learning_rate = 1e-4
	total_time = 0

	# マルチプロセスでデータを準備する場合
	if args.multiprocessing:
		num_preloads = 30
		queue = preloading_loop(dataset, augmentation, num_preloads, Queue())
		preloading_process = None

	# 学習ループ
	total_iterations_train = dataset.get_total_training_iterations()
	for epoch in xrange(1, args.total_epoch + 1):
		print_bold("Epoch %d" % epoch)
		start_time = time.time()
		loss_value = 0
		sum_loss = 0
		
		with chainer.using_config("train", True):
			current_iteration = 0

			while total_iterations_train > current_iteration:

				if args.multiprocessing:
					minibatch_list = []
					for _ in range(num_preloads):
						minibatch_list.append(queue.get())

					if preloading_process is not None:
						preloading_process.join()

					queue = Queue()
					preloading_process = Process(target=preloading_loop, args=(dataset, augmentation, num_preloads, queue))
					preloading_process.start()
				else:
					minibatch_list = [dataset.get_minibatch(option=augmentation, gpu=False)]

				for batch_idx, data in enumerate(minibatch_list):
					try:
						x_batch, x_length_batch, t_batch, t_length_batch, bucket_idx, group_idx = data

						if args.gpu_device >= 0:
							x_batch = cuda.to_gpu(x_batch.astype(np.float32))
							t_batch = cuda.to_gpu(t_batch.astype(np.int32))
							x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
							t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

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
					sys.stdout.write("\riteration {}/{}".format(batch_idx + current_iteration + 1, total_iterations_train))
					sys.stdout.flush()

				current_iteration += len(minibatch_list)

		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()
		save_model(args.model_dir, model)

		# バリデーション
		with chainer.using_config("train", False):
			# ノイズ無しデータでバリデーション
			iterator = dataset.get_iterator_dev(batchsizes, None, gpu=args.gpu_device >= 0)
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
	parser.add_argument("--multiprocessing", "-multi", default=False, action="store_true")
	
	parser.add_argument("--ndim-audio-features", "-features", type=int, default=3)
	parser.add_argument("--ndim-h", "-dh", type=int, default=128)
	parser.add_argument("--ndim-dense", "-dd", type=int, default=320)
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
