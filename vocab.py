# coding: utf", "8
from __future__ import print_function
import os, codecs, sys
import jaconv

# 参考文献
# [音声認識のための音響モデルと言語モデルの仕様](http://pj.ninjal.ac.jp/corpus_center/csj/manu-f/asr.pdf)
# [トライフォンの扱いとHMMListについて](http://julius.osdn.jp/index.php?q=doc/triphone.html)
# [音韻モデルの作成](http://winnie.kuis.kyoto-u.ac.jp/dictation/doc/phone_m.pdf)

VOWELS = ["a", "i", "u", "e", "o"]

PHONEMES_EXCEPT_VOWEL = [
	"N", "w", "y", "j", "my", "ky", "by", "gy", "ny", "hy", "ry", "py", "p", "t", "k", "ts", "ch", 
	"b", "d", "g", "z", "m", "n", "s", "sh", "h", "f", "r", "q"
]

KATAKANA_PHONEME = {
	"ア": ["a"],
	"イ": ["i"],
	"ウ": ["u"],
	"エ": ["e"],
	"オ": ["o"],
	"カ": ["k", "a"],
	"キ": ["k", "i"],
	"ク": ["k", "u"],
	"ケ": ["k", "e"],
	"コ": ["k", "o"],
	"ガ": ["g", "a"],
	"ギ": ["g", "i"],
	"グ": ["g", "u"],
	"ゲ": ["g", "e"],
	"ゴ": ["g", "o"],
	"サ": ["s", "a"],
	"シ": ["sh", "i"],
	"ス": ["s", "u"],
	"セ": ["s", "e"],
	"ソ": ["s", "o"],
	"ザ": ["z", "a"],
	"ジ": ["j", "i"],
	"ズ": ["z", "u"],
	"ゼ": ["z", "e"],
	"ゾ": ["z", "o"],
	"タ": ["t", "a"],
	"チ": ["ch", "i"],
	"ツ": ["ts", "u"],
	"テ": ["t", "e"],
	"ト": ["t", "o"],
	"ダ": ["d", "a"],
	"ヂ": ["j", "i"],
	"ヅ": ["z", "u"],
	"デ": ["d", "e"],
	"ド": ["d", "o"],
	"ナ": ["n", "a"],
	"ニ": ["n", "i"],
	"ヌ": ["n", "u"],
	"ネ": ["n", "e"],
	"ノ": ["n", "o"],
	"ハ": ["h", "a"],
	"ヒ": ["h", "i"],
	"フ": ["h", "u"],
	"ヘ": ["h", "e"],
	"ホ": ["h", "o"],
	"バ": ["b", "a"],
	"ビ": ["b", "i"],
	"ブ": ["b", "u"],
	"ベ": ["b", "e"],
	"ボ": ["b", "o"],
	"パ": ["p", "a"],
	"ピ": ["p", "i"],
	"プ": ["p", "u"],
	"ペ": ["p", "e"],
	"ポ": ["p", "o"],
	"マ": ["m", "a"],
	"ミ": ["m", "i"],
	"ム": ["m", "u"],
	"メ": ["m", "e"],
	"モ": ["m", "o"],
	"ラ": ["r", "a"],
	"リ": ["r", "i"],
	"ル": ["r", "u"],
	"レ": ["r", "e"],
	"ロ": ["r", "o"],
	"ワ": ["w", "a"],
	"ヲ": ["o"],
	"ヤ": ["y", "a"],
	"ユ": ["y", "u"],
	"ヨ": ["y", "o"],
	"キャ": ["ky", "a"],
	"キュ": ["ky", "u"],
	"キョ": ["ky", "o"],
	"ギャ": ["gy", "a"],
	"ギュ": ["gy", "u"],
	"ギョ": ["gy", "o"],
	"シャ": ["sh", "a"],
	"シュ": ["sh", "u"],
	"ショ": ["sh", "o"],
	"ジャ": ["j", "a"],
	"ジュ": ["j", "u"],
	"ジョ": ["j", "o"],
	"チャ": ["ch", "a"],
	"チュ": ["ch", "u"],
	"チョ": ["ch", "o"],
	"ニャ": ["ny", "a"],
	"ニュ": ["ny", "u"],
	"ニョ": ["ny", "o"],
	"ヒャ": ["hy", "a"],
	"ヒュ": ["hy", "u"],
	"ヒョ": ["hy", "o"],
	"ビャ": ["by", "a"],
	"ビュ": ["by", "u"],
	"ビョ": ["by", "o"],
	"ピャ": ["py", "a"],
	"ピュ": ["py", "u"],
	"ピョ": ["py", "o"],
	"ミャ": ["my", "a"],
	"ミュ": ["my", "u"],
	"ミョ": ["my", "o"],
	"リャ": ["ry", "a"],
	"リュ": ["ry", "u"],
	"リョ": ["ry", "o"],
	"イェ": ["i", "e"],
	"シェ": ["sh", "e"],
	"ジェ": ["j", "e"],
	"ティ": ["t", "i"],
	"トゥ": ["t", "u"],
	"チェ": ["ch", "e"],
	"ツァ": ["ts", "a"],
	"ツィ": ["ts", "i"],
	"ツェ": ["ts", "e"],
	"ツォ": ["ts", "o"],
	"ディ": ["d", "i"],
	"ドゥ": ["d", "u"],
	"デュ": ["d", "u"],
	"ニェ": ["n", "i", "e"],
	"ヒェ": ["h", "e"],
	"ファ": ["f", "a"],
	"フィ": ["f", "i"],
	"フェ": ["f", "e"],
	"フォ": ["f", "o"],
	"フュ": ["hy", "u"],
	"ブィ": ["b", "u", "i"],
	"ミェ": ["m" , "i", "e"],
	"ウィ": ["w", "i"],
	"ウェ": ["w", "e"],
	"ウォ": ["w", "o"],
	"クヮ": ["k", "a"],
	"グヮ": ["g", "a"],
	"スィ": ["s", "u", "i"],
	"ズィ": ["j", "i"],
	"テュ": ["ch", "u"],
	"ヴァ": ["b", "a"],
	"ヴィ": ["b", "i"],
	"ヴ": ["b", "u"],
	"ヴェ": ["b", "e"],
	"ヴォ": ["b", "o"],
	"ン": ["N"],
	"ッ": ["q"],
}

SUTEGANA_PHONEME = {
	"ァ": ["a"],
	"ィ": ["i"],
	"ゥ": ["u"],
	"ェ": ["e"],
	"ォ": ["o"],
	"ャ": ["a"],
	"ュ": ["u"],
	"ョ": ["o"],
	"ヮ": ["a"],
}

def merge(sequence):
	string = ""
	for phoneme in sequence:
		string += phoneme
	return string


def get_all_triphone():
	all_triphone = set()

	# 状態共有トライフォンにより状態数の削減
	shared_phoneme = [
		"N","a","a:","b","by","ch","d","dy","e","e:","f","g","gy","p",
		"py","q","r","ry","s","sh","t","ts","u","u:","w","y","z", "i", "o"
	]

	for katakana_L, phoneme_L in KATAKANA_PHONEME.items():	
		for katakana_X, phoneme_X in KATAKANA_PHONEME.items():
			for katakana_R, phoneme_R in KATAKANA_PHONEME.items():
				phoneme_sequence = phoneme_L + phoneme_X + phoneme_R
				triphone_candidates = []
				for L, X, R in zip([None] + phoneme_sequence[:-1], phoneme_sequence, phoneme_sequence[1:] + [None]):
					triphone_candidates.append(shared_triphonize(L, X, R, shared_phoneme))
				for triphone in triphone_candidates:
					all_triphone.add(triphone)

	return all_triphone


def convert_character_pair_to_phoneme(char_left, char_right):
	if char_right in SUTEGANA_PHONEME:
		return KATAKANA_PHONEME[char_left + char_right][:-1]
	if char_left in SUTEGANA_PHONEME:
		phoneme = SUTEGANA_PHONEME[char_left][:]
		if char_right == u"ー":
			phoneme[-1] += ":"
		return phoneme
	if char_left == u"ー":
		return []
	phoneme = KATAKANA_PHONEME[char_left][:]
	if char_right == u"ー":
		phoneme[-1] += ":"
	return phoneme

def convert_sentence_to_phoneme_sequence(sentence):
	sentence = jaconv.hira2kata(sentence)	# カタカナに変換
	phoneme_sequence = []
	for char_left, char_right in zip(sentence, sentence[1:] + " "):
		phoneme = convert_character_pair_to_phoneme(char_left, char_right)
		phoneme_sequence += phoneme
	return phoneme_sequence

def triphonize(L, X, R):
	triphone = ""
	if L is not None:
		triphone = L + "-"
	triphone += X
	if R is not None:
		triphone += "+" + R
	return triphone

def untriphonize(triphone):
	X = triphone.split("-")
	L, X = X if len(X) == 2 else (None, X[0])
	X = X.split("+")
	X, R = X if len(X) == 2 else (X[0], None)
	return L, X, R

def shared_triphonize(L, X, R, shared_phoneme):
	if X not in shared_phoneme:
		return triphonize(L, X, R)
	return X

def convert_phoneme_sequence_to_triphone_sequence(phoneme_sequence, convert_to_str=True):
	triphone_sequence = []
	for L, X, R in zip([None] + phoneme_sequence[:-1], phoneme_sequence, phoneme_sequence[1:] + [None]):
		if convert_to_str:
			triphone_sequence.append(triphonize(L, X, R))
		else:
			triphone_sequence.append((L, X, R))
	return triphone_sequence

def main():
	def shared_triphonize(L, X, R):
		if L:
			if L[-1] == ":":
				L = L[:-1]
		return triphonize(L, X, R)

	triphone_counts = {}
	trn_base_dir = "/home/stark/sandbox/CSJ_/"
	trn_dir_list = [os.path.join(trn_base_dir, category) for category in ["core", "noncore"]]
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
					for triphone in triphone_sequence:
						L, X, R = triphone
						if L is None or R is None:
							continue
						if X[-1] == ":":
							continue
						if X[0] in VOWELS:
							continue
						# 縮約
						# 文脈において長母音と通常の母音との違いを無視する
						if R and R[-1] == ":":
							R = R[:-1]
						if L and L[-1] == ":":
							L = L[:-1]
						# 右音素文脈では拗音を区別しない
						if R and len(R) == 2 and R[-1] == "y":
							R = R[:-1]
						# 拗音の左音素文脈を共通化する
						if L and len(L) == 2 and L[-1] == "y":
							L = "y"

						triphone_str = shared_triphonize(L, X, R)
						if triphone_str not in triphone_counts:
							triphone_counts[triphone_str] = 0
						triphone_counts[triphone_str] += 1
	accepted_triphones = []
	threshold = 1000
	count = 0
	for k, v in sorted(triphone_counts.items(), key=lambda x:x[1]):
		if v > threshold:
			count += 1
			accepted_triphones.append(untriphonize(k))
			# print(k, v)
	# print(count, len(triphone_counts))

	sorted_triphones = {}
	for triphone in accepted_triphones:
		L, X, R = triphone
		if X not in sorted_triphones:
			sorted_triphones[X] = set()
		sorted_triphones[X].add(triphonize(L, X, R))

	# 単語の開始
	for katakana, phoneme in KATAKANA_PHONEME.items():
		if len(phoneme) > 1:
			L, X, R = None, phoneme[0], phoneme[1]
			if X not in sorted_triphones:
				sorted_triphones[X] = set()
			sorted_triphones[X].add(triphonize(L, X, R))

	for vowel in VOWELS:
		for phoneme in PHONEMES_EXCEPT_VOWEL:
			L, X, R = None, vowel, phoneme[0]
			if X not in sorted_triphones:
				sorted_triphones[X] = set()
			sorted_triphones[X].add(triphonize(L, X, R))

	# 単語の終端

	total = 0
	for phoneme, triphones in sorted_triphones.items():
		total += len(triphone)
		for triphone in triphones:
			print(triphone)
	print(total)

if __name__ == "__main__":
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
	main()