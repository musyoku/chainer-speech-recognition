import sys, cupy, os
import chainer
import numpy as np
import chainer.functions as F
from chainer import cuda
from args import args
from model import Model
from asr.model.cnn import configure
from asr.error import compute_minibatch_error
from asr.data import AugmentationOption
from asr.data.loaders.buckets import Loader
from asr.utils import printb, printr, printc, bold
from asr.optimizers import get_learning_rate, decay_learning_rate, get_optimizer, set_learning_rate, set_momentum
from asr.vocab import get_unigram_ids, ID_BLANK
from asr.training import Environment, Iteration
from asr.logging import Report

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

	# 各種パス
	config_filename = os.path.join(args.working_directory, "config.json")
	model_filename = os.path.join(args.working_directory, "model.hdf5")
	env_filename = os.path.join(args.working_directory, "training.json")
	log_filename = os.path.join(args.working_directory, "log.txt")
	stats_directory = os.path.join(args.working_directory, "stats")

	# データの読み込み
	vocab_token_ids, vocab_id_tokens = get_unigram_ids()
	vocab_size = len(vocab_token_ids)

	# 設定
	config = configure()
	config.vocab_size = vocab_size
	config.ndim_rnn = args.ndim_rnn
	config.ndim_conv = args.ndim_conv
	config.ndim_dense = args.ndim_dense
	config.num_rnn_layers = args.num_rnn_layers
	config.kernel_size = (3, 5)
	config.dropout = args.dropout
	config.sampling_rate = args.sampling_rate
	config.frame_width = args.frame_width
	config.frame_shift = args.frame_shift
	config.num_mel_filters = args.num_mel_filters
	config.window_func = args.window_func
	config.using_delta = args.using_delta
	config.using_delta_delta = args.using_delta_delta
	config.ndim_audio_features = 1
	if args.using_delta:
		config.ndim_audio_features += 1
	if args.using_delta_delta:
		config.ndim_audio_features += 1
	config.save(config_filename)

	# バケツごとのミニバッチサイズ
	# 自動的に調整される
	batchsizes_train = [128] * 30
	batchsizes_dev = [size * 3 for size in batchsizes_train]

	# データセットの読み込み
	loader = Loader(
		data_path=args.dataset_path, 				# バケツ変換済みのデータへのパス
		batchsizes_train=batchsizes_train, 			# 学習時のバッチサイズ
		batchsizes_dev=batchsizes_dev, 				# 評価時のバッチサイズ
		buckets_limit=args.buckets_limit, 			# 用いるバケツの制限
		bucket_split_sec=0.5,						# バケツに区切る間隔
		buckets_cache_size=1000, 					# ディスクから読み込んだデータをメモリにキャッシュする個数
		vocab_token_to_id=vocab_token_ids,			# ユニグラム文字をIDに変換する辞書
		dev_split=0.01,								# データの何％を評価に用いるか
		seed=0,										# シード
		id_blank=ID_BLANK,							# ブランクのID
		apply_cmn=args.apply_cmn,					# ケプストラム平均正規化を使うかどうか。データセットに合わせる必要がある。
		sampling_rate=config.sampling_rate,			# サンプリングレート
		frame_width=config.frame_width,				# フレーム幅
		frame_shift=config.frame_shift,				# フレーム感覚
		num_mel_filters=config.num_mel_filters,		# フィルタバンクの数
		window_func=config.window_func,				# 窓関数
		using_delta=config.using_delta,				# Δ特徴量
		using_delta_delta=config.using_delta_delta	# ΔΔ特徴量
	)

	# データ拡大
	augmentation = AugmentationOption()
	if args.augmentation:
		augmentation.change_vocal_tract = True
		augmentation.change_speech_rate = True
		augmentation.add_noise = True

	# モデル
	model = Model(config)
	model.load(model_filename)

	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu(args.gpu_device)

	xp = model.xp
	using_gpu = False if xp is np else True

	# 環境
	def signal_handler():
		set_learning_rate(optimizer, env.learning_rate)
		set_momentum(optimizer, env.momentum)
		augmentation.change_vocal_tract = env.augmentation.change_vocal_tract
		augmentation.change_speech_rate = env.augmentation.change_speech_rate
		augmentation.add_noise = env.augmentation.add_noise
		printr("")
		env.dump()

	env = Environment(env_filename, signal_handler)
	env.learning_rate = args.learning_rate
	env.momentum = args.momentum
	env.final_learning_rate = args.final_learning_rate
	env.lr_decay = args.lr_decay
	env.augmentation = augmentation
	env.save()

	pid = os.getpid()
	print("Run '{}' to update training environment.".format(bold("kill -USR1 {}".format(pid))))

	# ログ
	loader.dump()
	env.dump()
	config.dump()

	# optimizer
	optimizer = get_optimizer(args.optimizer, env.learning_rate, args.momentum)
	optimizer.setup(model)
	if args.grad_clip > 0:
		optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	if args.weight_decay > 0:
		optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

	# データセットの平均・分散を推定
	if loader.load_stats(stats_directory) is False:
		print("Estimating the mean and unbiased variance of the dataset ...")
		loader.update_stats(20, [128] * 30, augmentation)
		loader.save_stats(stats_directory)

	# バッチサイズの調整
	print("Searching for the best batch size ...")
	batch_iter_train = loader.get_training_batch_iterator(batchsizes_train, augmentation=augmentation, gpu=using_gpu)
	for _ in range(30):
		for x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_id in batch_iter_train:
			try:
				with chainer.using_config("train", True):
					model.reset_state()
					loss = F.connectionist_temporal_classification(model(x_batch), t_batch, ID_BLANK, x_length_batch, t_length_batch)
			except Exception as e:
				if isinstance(e, cupy.cuda.runtime.CUDARuntimeError):
					batchsizes_train[bucket_id] = max(batchsizes_train[bucket_id] - 16, 4)
					print("new batchsize {} for bucket {}".format(batchsizes_train[bucket_id], bucket_id + 1))
			break
	batchsizes_dev = [size * 3 for size in batchsizes_train]

	# 学習
	printb("[Training]")
	epochs = Iteration(args.epochs)
	report = Report(log_filename)
	
	for epoch in epochs:
		sum_loss = 0

		# パラメータの更新
		batch_iter_train = loader.get_training_batch_iterator(batchsizes_train, augmentation=augmentation, gpu=using_gpu)
		total_iterations_train = batch_iter_train.get_total_iterations()

		for batch_index, (x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_id) in enumerate(batch_iter_train):

			try:
				with chainer.using_config("train", True):
					# print(xp.mean(x_batch, axis=3), xp.var(x_batch, axis=3))
					printr("iteration {}/{}".format(batch_index + 1, total_iterations_train))

					# 誤差の計算
					model.reset_state()
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

			except Exception as e:
				printr("")
				printc("{} (bucket {})".format(str(e), bucket_id + 1), color="red")
				if isinstance(e, cupy.cuda.runtime.CUDARuntimeError):
					batchsizes_train[bucket_id] -= 16
					batchsizes_train[bucket_id] = max(batchsizes_train[bucket_id], 4)
					batchsizes_dev = [size * 3 for size in batchsizes_train]
					print("new batchsize {} for bucket {}".format(batchsizes_train[bucket_id], bucket_id + 1))

		model.save(model_filename)
		loader.save_stats(stats_directory)

		report("Epoch {}".format(epoch))
		report(loader.get_statistics())

		# ノイズ無しデータでバリデーション
		batch_iter_dev = loader.get_development_batch_iterator(batchsizes_dev, augmentation=AugmentationOption(), gpu=using_gpu)
		total_iterations_dev = batch_iter_dev.get_total_iterations()
		buckets_errors = [[] for i in range(loader.get_num_buckets())]

		for batch_index, (x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_id) in enumerate(batch_iter_dev):

			try:
				with chainer.using_config("train", False):
					with chainer.no_backprop_mode():
						# print(xp.mean(x_batch, axis=3), xp.var(x_batch, axis=3))
						printr("Computing CER ... {}/{}".format(batch_index + 1, total_iterations_dev))
						y_batch = model(x_batch, split_into_variables=False)
						y_batch = xp.argmax(y_batch.data, axis=2)
						error = compute_minibatch_error(y_batch, t_batch, ID_BLANK, vocab_token_ids, vocab_id_tokens)
						buckets_errors[bucket_id].append(error)

			except Exception as e:
				printr("")
				printc("{} (bucket {})".format(str(e), bucket_id + 1), color="red")
				
		avg_errors_dev = []
		for errors in buckets_errors:
			avg_errors_dev.append(sum(errors) / len(errors) * 100)

		log_body = {
			"loss": sum_loss / batch_iter_train.get_total_iterations(),
			"CER": formatted_error(avg_errors_dev),
			"lr": get_learning_rate(optimizer)
		}
		epochs.console_log(log_body)
		report(log_body)

		# 学習率の減衰
		decay_learning_rate(optimizer, env.lr_decay, env.final_learning_rate)

if __name__ == "__main__":
	main()
