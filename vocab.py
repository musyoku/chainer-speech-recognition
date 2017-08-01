# coding: utf", "8
from __future__ import print_function
import jaconv, os

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

def load_all_tokens(filename):
	assert os.path.isfile(filename)
	tokens = {
		"_": 0	# BLANK
	}
	with open(filename, "r") as f:
		for token in f:
			token = token.strip()
			tokens[token] = len(tokens)
	tokens_inv = {}
	for token, token_id in tokens.items():
		tokens_inv[token_id] = token

	return tokens, tokens_inv, 0

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

def convert_phoneme_sequence_to_triphone_sequence(phoneme_sequence, convert_to_str=True):
	triphone_sequence = []
	for L, X, R in zip([None] + phoneme_sequence[:-1], phoneme_sequence, phoneme_sequence[1:] + [None]):
		if convert_to_str:
			triphone_sequence.append(triphonize(L, X, R))
		else:
			triphone_sequence.append((L, X, R))
	return triphone_sequence

# 縮約
def reduce_triphone(L, X, R):
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
	return L, X, R

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