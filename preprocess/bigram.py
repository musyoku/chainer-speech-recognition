# coding: utf", "8
from __future__ import print_function
import jaconv, os, sys, argparse, codecs
sys.path.append("../")
from vocab import get_all_bigram_tokens, convert_sentence_to_bigram_tokens

def main():
	bigram_counts = {}
	bigram_tokens = get_all_bigram_tokens()
	for (first, second) in bigram_tokens:
		bigram_counts[first + "+" + second] = 0
	trn_base_dir = "/home/stark/sandbox/CSJ_/"
	trn_dir_list = [os.path.join(trn_base_dir, category) for category in ["core", "noncore"]]
	all_triphone_sequences = []
	for dir_idx, trn_dir in enumerate(trn_dir_list):
		trn_files = os.listdir(trn_dir)
		for file_idx, trn_filename in enumerate(trn_files):
			trn_path = os.path.join(trn_dir, trn_filename)
			sys.stdout.write("\r{}/{} ({}/{})".format(file_idx + 1, len(trn_files), dir_idx + 1, len(trn_dir_list)))
			sys.stdout.flush()
			with codecs.open(trn_path, "r", "utf-8") as f:
				for data in f:
					components = data.split(":")
					assert len(components) == 3
					sentence = components[-1].strip()
					unigram_tokens = convert_sentence_to_bigram_tokens(sentence)
					if len(unigram_tokens) == 1:
						continue
					for first, second in zip(unigram_tokens[:-1], unigram_tokens[1:]):
						if first == u"ー":
							continue
						if second == u"ー":
							continue
						key = first + "+" + second
						if key not in bigram_counts:
							raise Exception(key)
						bigram_counts[key] += 1
	print(bigram_counts)
	threshold = 500
	accepted_bigram = []
	for token, count in sorted(bigram_counts.items(), key=lambda x:x[1]):
		if count < threshold:
			continue
		print(token, count)
		accepted_bigram.append(token)

	with open(args.output_filename, "w") as f:
		f.write("\n".join(accepted_bigram))

	print("total:", len(accepted_bigram))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--output-filename", "-file", type=str, default="../bigram.list") 
	args = parser.parse_args()
	main()