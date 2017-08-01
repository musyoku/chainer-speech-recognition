# coding: utf", "8
from __future__ import print_function
import jaconv, os, sys
sys.path.append("../")
from vocab import VOWELS, PHONEMES_EXCEPT_VOWEL, KATAKANA_PHONEME, SUTEGANA_PHONEME, convert_sentence_to_phoneme_sequence, convert_phoneme_sequence_to_triphone_sequence, untriphonize, triphonize, reduce_triphone

def main():
	diphone_counts = {}
	triphone_counts = {}
	trn_base_dir = "/home/aibo/sandbox/CSJ_/"
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
					phoneme_sequence = convert_sentence_to_phoneme_sequence(sentence)
					triphone_sequence = convert_phoneme_sequence_to_triphone_sequence(phoneme_sequence, False)
					all_triphone_sequences.append(triphone_sequence)
					for triphone in triphone_sequence:
						L, X, R = triphone
						if X[-1] == ":":
							continue
						if X[0] in VOWELS:
							continue

						L, X, R = reduce_triphone(L, X, R)

						if L is None or R is None:
							diphone_str = triphonize(L, X, R)
							if diphone_str not in diphone_counts:
								diphone_counts[diphone_str] = 0
							diphone_counts[diphone_str] += 1
							continue
							
						triphone_str = triphonize(L, X, R)
						if triphone_str not in triphone_counts:
							triphone_counts[triphone_str] = 0
						triphone_counts[triphone_str] += 1
	accepted_triphones = []
	threshold_tri = 1000
	count = 0
	for token, count in sorted(triphone_counts.items(), key=lambda x:x[1]):
		if count > threshold_tri:
			print(token, count)
			count += 1
			accepted_triphones.append(untriphonize(token))

	sorted_triphones = {}
	for triphone in accepted_triphones:
		L, X, R = triphone
		if X not in sorted_triphones:
			sorted_triphones[X] = set()
		sorted_triphones[X].add(triphonize(L, X, R))

	# 単語の開始・終端のトライフォンを追加
	threshold_di = 100
	for token, count in sorted(diphone_counts.items(), key=lambda x:x[1]):
		if count > threshold_di:
			L, X, R = untriphonize(token)
			if X not in sorted_triphones:
				sorted_triphones[X] = set()
			sorted_triphones[X].add(triphonize(L, X, R))
			print(token, count)

	# モノフォンを追加
	for katakana, phoneme in KATAKANA_PHONEME.items():
		X = phoneme[0]
		if X not in sorted_triphones:
			sorted_triphones[X] = set()
		sorted_triphones[X].add(triphonize(None, X, None))
	for phoneme in VOWELS + ["N"]:
		X = phoneme + ":"
		if X not in sorted_triphones:
			sorted_triphones[X] = set()
		sorted_triphones[X].add(triphonize(None, X, None))

	total = 0
	accepted_triphones = []
	for phoneme in sorted(sorted_triphones.keys()):
		triphones = sorted_triphones[phoneme]
		total += len(triphones)
		for token in triphones:
			accepted_triphones.append(token)

	print("total:", total)

	# 得られたトライフォン、ダイフォン、モノフォンがデータにどの程度出現するか
	assigned_counts = {}
	for token in accepted_triphones:
		assigned_counts[token] = 0

	def increment(triphone):
		L, X, R = triphone
		token = triphonize(L, X, R)
		if token in assigned_counts:
			assigned_counts[token] += 1
			return True
		return False

	for triphone_sequence in all_triphone_sequences:
		for triphone in triphone_sequence:
			L, X, R = triphone
			L, X, R = reduce_triphone(L, X, R)
			if increment((L, X, R)):
				continue
			if L and R:
				if increment((None, X, R)):
					continue
			if increment((None, X, None)):
				continue
			raise Exception(L, X, R)

	for token, count in sorted(assigned_counts.items(), key=lambda x:x[1]):
		if count == 0:
			accepted_triphones.remove(token)	# データに一度も出てこないモノフォンを削除
		print(token, count)

	with open(args.filename, "w") as f:
		f.write("\n".join(accepted_triphones))

	print("total:", len(accepted_triphones))

if __name__ == "__main__":
	import codecs, sys, argparse
	# sentence = u"けものフレンズさいしゅーかい"
	# sentence = u"こんかいけんとーしたしゅほーおはっぴょーさしていただきます"
	# phoneme_sequence = convert_sentence_to_phoneme_sequence(sentence)
	# print(phoneme_sequence)
	# triphone_sequence = convert_phoneme_sequence_to_triphone_sequence(phoneme_sequence)
	# print(triphone_sequence)
	# all_triphone = get_all_triphone()
	# print(all_triphone)
	# print(len(all_triphone))
	# raise Exception()

	parser = argparse.ArgumentParser()
	parser.add_argument("--filename", "-file", type=str, default="../triphone.list") 
	args = parser.parse_args()
	main()