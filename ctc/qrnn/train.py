# coding: utf8
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import sys, argparse
sys.path.append("../../")
import dataset

def main(args):
	dataset.load_wav_and_transcription()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-dir", "-model", type=str, default="model")
	args = parser.parse_args()
	main(args)