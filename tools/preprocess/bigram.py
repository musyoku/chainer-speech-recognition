import jaconv, os, sys, argparse, codecs
sys.path.append(os.path.join("..", ".."))
from asr.vocab import get_all_bigram_tokens, convert_sentence_to_unigram_tokens, UNIGRAM_TOKENS
from asr.utils import printr, printc

def main():
	unigram_counts = {}
	for token in UNIGRAM_TOKENS:
		unigram_counts[token] = 0

	bigram_counts = {}
	bigram_tokens = get_all_bigram_tokens()
	for (first, second) in bigram_tokens:
		bigram_counts[first + second] = 0

	trn_base_dir = "/home/stark/sandbox/CSJ_/"	# 変換済みの書き起こし
	trn_dir_list = [os.path.join(trn_base_dir, category) for category in ["core", "noncore"]]
	all_triphone_sequences = []

	for dir_idx, trn_dir in enumerate(trn_dir_list):
		trn_files = os.listdir(trn_dir)

		for file_idx, trn_filename in enumerate(trn_files):
			printr("\r{}/{} ({}/{})".format(file_idx + 1, len(trn_files), dir_idx + 1, len(trn_dir_list)))
			trn_path = os.path.join(trn_dir, trn_filename)

			with codecs.open(trn_path, "r", "utf-8") as f:
				for data in f:
					components = data.split(":")
					assert len(components) == 3
					sentence = components[-1].strip()
					unigram_tokens = convert_sentence_to_unigram_tokens(sentence)
					for token in unigram_tokens:
						if token not in unigram_counts:
							raise Exception(token)
						unigram_counts[token] += 1
					if len(unigram_tokens) == 1:
						continue
					for first, second in zip(unigram_tokens[:-1], unigram_tokens[1:]):
						if first == u"ー":
							continue
						if second == u"ー":
							continue
						key = first + second
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


	for token, count in sorted(unigram_counts.items(), key=lambda x:x[1]):
		if count < 1000:
			print(token, count)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--output-filename", "-file", type=str, default="../../bigram.list") 
	args = parser.parse_args()
	main()