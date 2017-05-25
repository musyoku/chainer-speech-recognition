# coding: utf8
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import sys, argparse
sys.path.append("../../")
from dataset import load_audio_and_transcription

def main(args):
	wav_paths = [
		"/home/stark/sandbox/CSJ/WAV/core/",
	]

	transcription_paths = [
		"/home/stark/sandbox/CSJ_/core/",
	]

	sampling_rate = 16000
	pair = load_audio_and_transcription(wav_paths, transcription_paths)
	dataset = []
	for signal, sentence in pair:
		# 転記、対数メルフィルタバンク出力、Δ、ΔΔの順で並べる
		# データが多いことが想定されるので適宜読み込むようにする
		dataset.append((signal, sentence, None, None, None))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-dir", "-model", type=str, default="model")
	args = parser.parse_args()
	main(args)