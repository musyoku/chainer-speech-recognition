import cupy, os, chainer
import numpy as np
import chainer.functions as F
from chainer import cuda
from model import load_model, load_config
from args import args
from asr.error import compute_minibatch_error
from asr.data import AugmentationOption
from asr.data.loaders.audio import Loader
from asr.utils import printb, printr, printc, bold
from asr.vocab import get_unigram_ids, ID_BLANK

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

	# バケツごとのミニバッチサイズ
	# 自動的に調整される
	batchsizes = [64] * 30

	# モデル
	model, config = load_model(model_filename, config_filename)
	assert model is not None

	# テストデータの読み込み
	loader = Loader(
		wav_directory_list=[
			"/home/stark/sandbox/CSJ/WAV/test",
		], 
		transcription_directory_list=[
			"/home/stark/sandbox/CSJ_/test",
		], 
		mean_filename=os.path.join(args.dataset_path, "mean.npy"), 
		std_filename=os.path.join(args.dataset_path, "std.npy"), 
		batchsizes=batchsizes,
		buckets_limit=args.buckets_limit, 			# 用いるバケツの制限
		bucket_split_sec=0.5,						# バケツに区切る間隔
		vocab_token_to_id=vocab_token_ids,			# ユニグラム文字をIDに変換する辞書
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

	augmentation = AugmentationOption()
	if args.augmentation:
		augmentation.change_vocal_tract = True
		augmentation.change_speech_rate = True
		augmentation.add_noise = True

	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu(args.gpu_device)

	xp = model.xp
	using_gpu = False if xp is np else True

	# ログ
	loader.dump()

	printb("[Test]")
	batch_iter = loader.get_batch_iterator(batchsizes, augmentation=augmentation, gpu=using_gpu)
	buckets_errors = [[] for i in range(loader.get_num_buckets())]

	for x_batch, x_length_batch, t_batch, t_length_batch, bigram_batch, bucket_id in batch_iter:

		try:
			with chainer.no_backprop_mode():
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
