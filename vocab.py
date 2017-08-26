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
	"ダ", "ヂ", "ヅ", "デ", "ド", 
	"ナ", "ニ", "ヌ", "ネ", "ノ", 
	"ハ", "ヒ", "フ", "ヘ", "ホ", 
	"バ", "ビ", "ブ", "ベ", "ボ", 
	"パ", "ピ", "プ", "ペ", "ポ", 
	"マ", "ミ", "ム", "メ", "モ", 
	"ラ", "リ", "ル", "レ", "ロ", 
	"ワ", "ヲ", "ヤ", "ユ", "ヨ", 
	"キャ", "キュ", "キョ", "ギャ", 
	"ギュ", "ギョ", "シャ", "シュ", 
	"ショ", "ジャ", "ジュ", "ジョ", 
	"チャ", "チュ", "チョ", "ニャ", 
	"ニュ", "ニョ", "ヒャ", "ヒュ", 
	"ヒョ", "ビャ", "ビュ", "ビョ", 
	"ピャ", "ピュ", "ピョ", "ミャ", 
	"ミュ", "ミョ", "リャ", "リュ", 
	"リョ", "イェ", "シェ", "ジェ", 
	"ティ", "トゥ", "チェ", "ツァ", 
	"ツィ", "ツェ", "ツォ", "ディ", 
	"ドゥ", "デュ", "ニェ", "ヒェ", 
	"ファ", "フィ", "フェ", "フォ", 
	"フュ", "ブィ", "ミェ", "ウィ", 
	"ウェ", "ウォ", "クヮ", "グヮ", 
	"スィ", "ズィ", "テュ", "ヴァ", 
	"ヴィ", "ヴ", "ヴェ", "ヴォ", 
	"ン", "ッ"
]

SUTEGANA = [
	"ァ","ィ","ゥ","ェ","ォ","ャ","ュ","ョ","ヮ",
]

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

def convert_sentence_to_unigram_tokens(sentence):
	tokens = []
	for char in sentence:
		if char in SUTEGANA:
			assert len(tokens) > 0
			tokens[-1] += char
			continue
		tokens.append(char)
	return tokens
