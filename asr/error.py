# coding: utf-8
import sys
import numpy as np
from .utils import printb, printr, stdout
from .vocab import convert_sentence_to_unigram_ids

def compute_character_error_rate(r, h):
	if len(r) == 0:
		return len(h)
	d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
	for i in range(len(r) + 1):
		for j in range(len(h) + 1):
			if i == 0: d[0][j] = j
			elif j == 0: d[i][0] = i
	for i in range(1, len(r) + 1):
		for j in range(1, len(h) + 1):
			if r[i-1] == h[j-1]:
				d[i][j] = d[i-1][j-1]
			else:
				substitute = d[i-1][j-1] + 1
				insert = d[i][j-1] + 1
				delete = d[i-1][j] + 1
				d[i][j] = min(substitute, insert, delete)
	return float(d[len(r)][len(h)]) / len(r)

def compute_minibatch_error(y_batch, t_batch, BLANK, vocab_token_to_id, vocab_id_to_token, print_sequences=False):
	sum_error = 0

	if print_sequences:
		printr("")

	for batch_idx, (argmax_sequence, true_sequence) in enumerate(zip(y_batch, t_batch)):
		target_id_sequence = []
		for token_id in true_sequence:
			if token_id == BLANK:
				continue
			target_id_sequence.append(int(token_id))
		pred_id_seqence = []
		prev_token = BLANK
		for token_id in argmax_sequence:
			if token_id == BLANK:
				prev_token = BLANK
				continue
			if token_id == prev_token:
				continue
			pred_id_seqence.append(int(token_id))
			prev_token = token_id

		# 一旦ユニグラムの文字列に戻す
		pred_sentence = ""
		for token_id in pred_id_seqence:
			pred_sentence += vocab_id_to_token[token_id]
		pred_id_seqence = convert_sentence_to_unigram_ids(pred_sentence, vocab_token_to_id)

		sum_error += compute_character_error_rate(target_id_sequence, pred_id_seqence)

		if print_sequences and vocab_id_to_token is not None:
			print("#{}".format(batch_idx + 1))
			pred_str = ""
			for token_id in pred_id_seqence:
				pred_str += vocab_id_to_token[token_id]
			printb("pred:	" + pred_str)
			target_str = ""
			for token_id in target_id_sequence:
				target_str += vocab_id_to_token[token_id]
			print("true:	" + target_str)

	return sum_error / len(y_batch)
