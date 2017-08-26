# coding: utf", "8
from __future__ import print_function
import jaconv, os

BLANK = 0

UNIGRAM_TOKENS = [
	"ア", "イ", "ウ", "エ", "オ", 
	"カ", "キ", "ク", "ケ", "コ", 
	"ガ", "ギ", "グ", "ゲ", "ゴ", 
	"サ", "シ", "ス", "セ", "ソ", 
	"ザ", "ジ", "ズ", "ゼ", "ゾ", 
	"タ", "チ", "ツ", "テ", "ト", 
	"ダ", "デ", "ド", 
	"ナ", "ニ", "ヌ", "ネ", "ノ", 
	"ハ", "ヒ", "フ", "ヘ", "ホ", 
	"バ", "ビ", "ブ", "ベ", "ボ", 
	"パ", "ピ", "プ", "ペ", "ポ", 
	"マ", "ミ", "ム", "メ", "モ", 
	"ラ", "リ", "ル", "レ", "ロ", 
	"ワ", "ヤ", "ユ", "ヨ", 
	"キャ", "キュ", "キョ", "ギャ", 
	"ギュ", "ギョ", "シャ", "シュ", 
	"ショ", "ジャ", "ジュ", "ジョ", 
	"チャ", "チュ", "チョ", 
	"ニャ", "ニュ", "ニョ", 
	"ヒャ", "ヒュ", 	"ヒョ", 
	"ビャ", "ビュ", "ビョ", 
	"ピャ", "ピュ", "ピョ", 
	"ミャ", 	"ミュ", "ミョ", 
	"リャ", "リュ", 	"リョ", 
	"シェ", "ジェ", 
	"ティ", "トゥ", "チェ", 
	"ディ", 	"ドゥ", "デュ",
	"ファ", "フィ", "フェ", "フォ", 
	"ウィ", "ウェ", "ウォ", 
	"ン", "ッ", "ー",
]

SUTEGANA = [
	"ァ","ィ","ゥ","ェ","ォ","ャ","ュ","ョ","ヮ",
]

UNIGRAM_COLLAPSE = {
	"ヒェ": "ヒエ",
	"ミェ": "ミエ",
	"ヴァ": "バ",
	"ヴィ": "ビ",
	"ヴ": "ブ",
	"ヴェ": "ベ",
	"ヴォ": "ボ",
	"クヮ": "クワ",
	"グヮ": "グワ",
	"フュ": "ヒュ",
	"ニェ": "ニエ",
	"ツィ": "ツイ",
	"イェ": "イエ",
	"ズィ": "ズイ",
	"ツェ": "ツエ",
	"ツァ": "ツア",
	"ツォ": "ツオ",
	"スィ": "シ",
	"テュ": "テユ",
}

ID_BLANK = 0

def get_unigram_ids():
	ids = {
		"_": ID_BLANK	# blank
	}
	
	for char in UNIGRAM_TOKENS:
		ids[char] = len(ids)

	ids_inv = {}
	for token, token_id in ids.items():
		ids_inv[token_id] = token

	return ids, ids_inv, ID_BLANK

UNIGRAM_TOKEN_IDS, UNIGRAM_ID_TOKENS, _ = get_unigram_ids()

def load_all_bigram_ids(filename):
	assert os.path.isfile(filename)
	ids = {
		"_": ID_BLANK	# blank
	}

	with open(filename, "r") as f:
		for token in f:
			token = token.strip()
			ids[token] = len(ids)

	ids_inv = {}
	for token, token_id in ids.items():
		ids_inv[token_id] = token

	return ids, ids_inv, ID_BLANK

def get_all_bigram_tokens():
	tokens = []
	for first in UNIGRAM_TOKENS:
		for second in UNIGRAM_TOKENS:
			tokens.append((first, second))
	return tokens

def convert_sentence_to_unigram_ids(sentence):
	tokens = convert_sentence_to_unigram_tokens(sentence)
	ids = []
	for token in tokens:
		assert token in UNIGRAM_TOKEN_IDS
		ids.append(UNIGRAM_TOKEN_IDS[token])
	return ids

def convert_sentence_to_unigram_tokens(sentence):
	_tokens = []
	for char in sentence:
		if char in SUTEGANA:
			assert len(_tokens) > 0
			_tokens[-1] += char
			continue
		_tokens.append(char)
	for i, token in enumerate(_tokens):
		if token in UNIGRAM_COLLAPSE:
			_tokens[i] = UNIGRAM_COLLAPSE[token]
	tokens = []
	for token in _tokens:
		for char in token:
			if char in SUTEGANA:
				assert len(tokens) > 0
				tokens[-1] += char
				continue
			tokens.append(char)
	return tokens
