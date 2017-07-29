# coding: utf", "8
from __future__ import print_function
import jaconv

# 参考文献
# [音声認識のための音響モデルと言語モデルの仕様](http://pj.ninjal.ac.jp/corpus_center/csj/manu-f/asr.pdf)
# [トライフォンの扱いとHMMListについて](http://julius.osdn.jp/index.php?q=doc/triphone.html)
# [音韻モデルの作成](http://winnie.kuis.kyoto-u.ac.jp/dictation/doc/phone_m.pdf)

katakana_phoneme = {
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

sutegana_phoneme = {
	"ァ": ["a"],
	"ィ": ["i"],
	"ゥ": ["u"],
	"ェ": ["e"],
	"ォ": ["o"],
	"ャ": ["a"],
	"ュ": ["u"],
	"ョ": ["o"],
}

def merge(sequence):
	string = ""
	for phoneme in sequence:
		string += phoneme
	return string

def get_all_triphone():
	all_triphone = set()

	# 状態共有トライフォンにより状態数の削減
	shared_phoneme = ["N", "a", ]

	for katakana_L, phoneme_L in katakana_phoneme.items():	
		for katakana_X, phoneme_X in katakana_phoneme.items():
			for katakana_R, phoneme_R in katakana_phoneme.items():
				phoneme_sequence = phoneme_L + phoneme_X + phoneme_R
				triphone_candidates = []
				for L, X, R in zip([None] + phoneme_sequence[:-1], phoneme_sequence, phoneme_sequence[1:] + [None]):
					triphone_candidates.append(triphonize(L, X, R))
				for triphone in triphone_candidates:
					all_triphone.add(triphone)



	return all_triphone


def convert_character_pair_to_phoneme(char_left, char_right):
	if char_right in sutegana_phoneme:
		return katakana_phoneme[char_left + char_right][:-1]
	if char_left in sutegana_phoneme:
		phoneme = sutegana_phoneme[char_left]
		if char_right == u"ー":
			phoneme[-1] += ":"
		return phoneme
	if char_left == u"ー":
		return []
	return katakana_phoneme[char_left]

def convert_sentence_to_phoneme_sequence(sentence):
	sentence = jaconv.hira2kata(sentence)	# カタカナに変換
	phoneme_sequence = []
	for char_left, char_right in zip(sentence, sentence[1:] + " "):
		phoneme = convert_character_pair_to_phoneme(char_left, char_right)
		print(char_left, char_right, phoneme)
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

def convert_phoneme_sequence_to_triphone_sequence(phoneme_sequence):
	triphone_sequence = []
	for L, X, R in zip([None] + phoneme_sequence[:-1], phoneme_sequence, phoneme_sequence[1:] + [None]):
		triphone_sequence.append(triphonize(L, X, R))
	return triphone_sequence

if __name__ == "__main__":
	sentence = u"けものフレンズさいしゅーかい"
	phoneme_sequence = convert_sentence_to_phoneme_sequence(sentence)
	print(phoneme_sequence)
	triphone_sequence = convert_phoneme_sequence_to_triphone_sequence(phoneme_sequence)
	print(triphone_sequence)
	all_triphone = get_all_triphone()
	print(all_triphone)
	print(len(all_triphone))