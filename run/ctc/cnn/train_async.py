# coding: utf8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import sys, argparse, time, cupy, math, os, binascii, signal
import chainer, cupy
import numpy as np
import chainer.functions as F
from chainer import cuda
from multiprocessing import Process, Queue
from model import load_model, save_model, build_model, save_config
from args import args
from asr.error import compute_minibatch_error
from asr.dataset import Dataset, AugmentationOption
from asr.utils import stdout, printb, printr, bold, printc
from asr.optimizers import get_learning_rate, decay_learning_rate, get_optimizer, set_learning_rate
from asr.vocab import get_unigram_ids, ID_BLANK
from asr.training import Environment, Iteration
from asr.logging import Report

def formatted_error(error_values):
	errors = []
	for error in error_values:
		errors.append("%.2f" % error)
	return errors

def preloading_loop(dataset, augmentation, num_load, queue):
	# print("preloading started.")
	np.random.seed(int(binascii.hexlify(os.urandom(4)), 16))
	for i in range(num_load):
		queue.put(dataset.sample_minibatch(augmentation=augmentation, gpu=False))
	# print("preloading done.")
	return queue

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
	config.ndim_audio_features = args.ndim_audio_features
	config.ndim_h = args.ndim_h
	config.ndim_dense = args.ndim_dense
	config.num_conv_layers = args.num_conv_layers
	config.kernel_size = (3, 5)
	config.dropout = args.dropout
	config.weightnorm = args.weightnorm
	config.wgain = args.wgain
	config.architecture = args.architecture
	return config

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
	env_filename = os.path.join(args.working_directory, "training.json")
	log_filename = os.path.join(args.working_directory, "log.txt")

	# データの読み込み
	vocab_token_ids, vocab_id_tokens = get_unigram_ids()
	vocab_size = len(vocab_token_ids)

	# 設定
	config = configure()
	config.vocab_size = vocab_size
	save_config(config_filename, config)

	# バケツごとのミニバッチサイズ
	# 自動的に調整される
	batchsizes_train = [128] * 30
	batchsizes_dev = [size * 3 for size in batchsizes_train]

	dataset = Dataset(args.dataset_path, batchsizes_train, batchsizes_dev, args.buckets_limit, token_ids=vocab_token_ids, id_blank=ID_BLANK, 
		buckets_cache_size=0, apply_cmn=args.apply_cmn)

	augmentation = AugmentationOption()
	if args.augmentation:
		augmentation.change_vocal_tract = True
		augmentation.change_speech_rate = True
		augmentation.add_noise = True

	# モデル
	model = load_model(model_filename, config_filename)
	if model is None:
		model = build_model(vocab_size=vocab_size, ndim_audio_features=config.ndim_audio_features, 
		 ndim_h=config.ndim_h, ndim_dense=config.ndim_dense, num_conv_layers=config.num_conv_layers,
		 kernel_size=(3, 5), dropout=config.dropout, weightnorm=config.weightnorm, wgain=config.wgain,
		 num_mel_filters=config.num_mel_filters, architecture=config.architecture)

	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu(args.gpu_device)

	xp = model.xp
	using_gpu = False if xp is np else True

	# 環境
	def signal_handler():
		set_learning_rate(optimizer, env.learning_rate)
		augmentation.change_vocal_tract = env.augmentation.change_vocal_tract
		augmentation.change_speech_rate = env.augmentation.change_speech_rate
		augmentation.add_noise = env.augmentation.add_noise
		printr("")
		env.dump()

	env = Environment(env_filename, signal_handler)
	env.learning_rate = args.learning_rate
	env.final_learning_rate = args.final_learning_rate
	env.lr_decay = args.lr_decay
	env.augmentation = augmentation
	env.save()

	pid = os.getpid()
	print("Run '{}' to update training environment.".format(bold("kill -USR1 {}".format(pid))))

	# ログ
	dataset.dump()
	env.dump()

	# optimizer
	optimizer = get_optimizer(args.optimizer, env.learning_rate, args.momentum)
	optimizer.setup(model)
	if args.grad_clip > 0:
		optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	if args.weight_decay > 0:
		optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

	# バッチサイズの調整
	print("Searching for the best batch size ...")
	itr = 0
	batch_train = dataset.get_training_batch_iterator(batchsizes_train, augmentation=augmentation, gpu=using_gpu)
	for x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_id, group_idx in batch_train:
		try:
			with chainer.using_config("train", True):
				y_batch = model(x_batch)
				loss = F.connectionist_temporal_classification(y_batch, t_batch, ID_BLANK, x_length_batch, t_length_batch)
		except Exception as e:
			printr("")
			printc("{} (bucket {})".format(str(e), bucket_id + 1), color="red")
			if isinstance(e, cupy.cuda.runtime.CUDARuntimeError):
				batchsizes_train[bucket_id] -= 16
				batchsizes_train[bucket_id] = max(batchsizes_train[bucket_id], 4)
				batchsizes_dev = [size * 3 for size in batchsizes_train]
				print("new batchsize {} for bucket {}".format(batchsizes_train[bucket_id], bucket_id + 1))
		itr += 1
		if itr > 50:
			break
			
	# マルチプロセスでデータの先読み
	num_preloads = 30
	queue = preloading_loop(dataset, augmentation, num_preloads, Queue())
	preloading_process = None

	# 学習
	printb("[Training]")
	epochs = Iteration(args.epochs)
	report = Report(log_filename)
	total_iterations_train = dataset.get_total_training_iterations()

	for epoch in epochs:
		sum_loss = 0

		# パラメータの更新
		current_iteration = 0
		while total_iterations_train > current_iteration:
			minibatch_list = []
			for _ in range(num_preloads):
				minibatch_list.append(queue.get())

			if preloading_process is not None:
				preloading_process.join()

			queue = Queue()
			preloading_process = Process(target=preloading_loop, args=(dataset, augmentation, num_preloads, queue))
			preloading_process.start()

			for batch_idx, data in enumerate(minibatch_list):
				try:
					with chainer.using_config("train", True):
						x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_id, group_idx = data

						if args.gpu_device >= 0:
							x_batch = cuda.to_gpu(x_batch.astype(np.float32))
							t_batch = cuda.to_gpu(t_batch.astype(np.int32))
							x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
							t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

						# print(xp.mean(x_batch, axis=3), xp.var(x_batch, axis=3))

						# 誤差の計算
						y_batch = model(x_batch)
						loss = F.connectionist_temporal_classification(y_batch, t_batch, ID_BLANK, x_length_batch, t_length_batch)

						# NaN
						loss_value = float(loss.data)
						if loss_value != loss_value:
							printc("Encountered NaN when computing loss.", color="red")
							continue

						# 更新
						optimizer.update(lossfun=lambda: loss)
						
						sum_loss += loss_value
						printr("iteration {}/{}".format(batch_idx + current_iteration + 1, total_iterations_train))

				except Exception as e:
					printr("")
					printc("{} (bucket {})".format(str(e), bucket_id + 1), color="red")
					if isinstance(e, cupy.cuda.runtime.CUDARuntimeError):
						batchsizes_train[bucket_id] -= 16
						batchsizes_train[bucket_id] = max(batchsizes_train[bucket_id], 4)
						batchsizes_dev = [size * 3 for size in batchsizes_train]
						print("new batchsize {} for bucket {}".format(batchsizes_train[bucket_id], bucket_id + 1))
						dataset.set_batchsizes_train(batchsizes_train)
						dataset.set_batchsizes_dev(batchsizes_dev)
						break	# 最初からやり直す

			current_iteration += len(minibatch_list)

		save_model(os.path.join(args.working_directory, "model.hdf5"), model)

		report("Epoch {}".format(epoch))
		report(dataset.get_statistics())

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

		log = {
			"loss": sum_loss / batch_train.get_total_iterations(),
			"CER": formatted_error(avg_errors_dev),
			"lr": get_learning_rate(optimizer)
		}
		epochs.log(log)
		report(log)

		# 学習率の減衰
		decay_learning_rate(optimizer, env.lr_decay, env.final_learning_rate)

def _main():
	assert args.dataset_path is not None
	# データの読み込み
	vocab_token_ids, vocab_id_tokens = get_unigram_ids()
	vocab_size = len(vocab_token_ids)

	# 設定
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
	config.vocab_size = vocab_size
	config.ndim_audio_features = args.ndim_audio_features
	config.ndim_h = args.ndim_h
	config.ndim_dense = args.ndim_dense
	config.num_conv_layers = args.num_conv_layers
	config.kernel_size = (3, 5)
	config.dropout = args.dropout
	config.weightnorm = args.weightnorm
	config.wgain = args.wgain
	config.architecture = args.architecture

	# バケツごとのミニバッチサイズ
	# GTX 1080 1台基準
	batchsizes = [32, 32, 32, 24, 16, 16, 12, 12, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

	cache_size = 0 if args.multiprocessing else 200	# マルチプロセスの場合キャッシュごとコピーされるのでメモリを圧迫する
	dataset = Dataset(args.dataset_path, batchsizes, args.buckets_limit, token_ids=vocab_token_ids, id_blank=ID_BLANK, 
		buckets_cache_size=cache_size, apply_cmn=args.apply_cmn)
	dataset.dump_information()

	augmentation = AugmentationOption()
	if args.augmentation:
		augmentation.change_vocal_tract = True
		augmentation.change_speech_rate = True
		augmentation.add_noise = True

	# モデル
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
	if args.grad_clip > 0:
		optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	if args.weight_decay > 0:
		optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
	final_learning_rate = 1e-6
	total_time = 0

	# マルチプロセスでデータを準備する場合
	if args.multiprocessing:
		num_preloads = 30
		queue = preloading_loop(dataset, augmentation, num_preloads, Queue())
		preloading_process = None

	# シグナルで学習率の制御
	pid = os.getpid()
	print("kill -USR1 {}".format(pid))

	def signal_usr1_handler(signum, stack):
		decay_learning_rate(optimizer, 0.1, final_learning_rate)
		printr("")
		print("new learning rate: {}".format(get_current_learning_rate(optimizer)))

	signal.signal(signal.SIGUSR1, signal_usr1_handler)

	# 学習ループ
	total_iterations_train = dataset.get_total_training_iterations()
	for epoch in xrange(1, args.total_epoch + 1):
		printb("Epoch %d" % epoch)
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
						x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx, group_idx = data

						# print(np.mean(x_batch, axis=3), np.var(x_batch, axis=3))

						if args.gpu_device >= 0:
							x_batch = cuda.to_gpu(x_batch.astype(np.float32))
							t_batch = cuda.to_gpu(t_batch.astype(np.int32))
							x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
							t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

						# 誤差の計算
						y_batch = model(x_batch)	# list of variables
						loss = F.connectionist_temporal_classification(y_batch, t_batch, ID_BLANK, x_length_batch, t_length_batch)

						# NaN
						loss_value = float(loss.data)
						if loss_value != loss_value:
							raise Exception("Encountered NaN when computing loss.")

						# 更新
						optimizer.update(lossfun=lambda: loss)

					except Exception as e:
						print(" ", bucket_idx, str(e))

					sum_loss += loss_value
					printr("iteration {}/{}".format(batch_idx + current_iteration + 1, total_iterations_train))

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
					x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_idx, group_idx = batch

					printr("computing CER of bucket {} (group {})".format(bucket_idx + 1, group_idx + 1))

					y_batch = model(x_batch, split_into_variables=False)
					y_batch = xp.argmax(y_batch.data, axis=2)
					error = compute_minibatch_error(y_batch, t_batch, ID_BLANK, vocab_token_ids, vocab_id_tokens)

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
	main()
