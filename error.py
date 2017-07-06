import numpy as np

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

def compute_minibatch_error(y_batch, t_batch, BLANK):
	sum_error = 0

	for argmax_sequence, true_sequence in zip(y_batch, t_batch):
		target_sequence = []
		for token in true_sequence:
			if token == BLANK:
				continue
			target_sequence.append(int(token))
		pred_seqence = []
		prev_token = BLANK
		for token in argmax_sequence:
			if token == BLANK:
				prev_token = BLANK
				continue
			if token == prev_token:
				continue
			pred_seqence.append(int(token))
			prev_token = token
		sum_error += compute_character_error_rate(target_sequence, pred_seqence)

	return sum_error / len(y_batch)

def compute_error(model, buckets_indices, buckets_feature, buckets_feature_length, buckets_sentence, buckets_batchsize, BLANK, mean_x_batch, stddev_x_batch, approximate=True):
	errors = []
	xp = model.xp
	for bucket_idx in range(len(buckets_indices)):
		data_indices = buckets_indices[bucket_idx]
		batchsize = buckets_batchsize[bucket_idx]
		feature_bucket = buckets_feature[bucket_idx]
		feature_length_bucket = buckets_feature_length[bucket_idx]
		sentence_bucket = buckets_sentence[bucket_idx]

		total_iterations = 1 if approximate else int(math.ceil(len(data_indices) / batchsize))

		if total_iterations == 1 and len(data_indices) < batchsize:
			batchsize = len(data_indices)

		sum_error = 0
		for itr in range(1, total_iterations + 1):

			x_batch, x_length_batch, t_batch, t_length_batch = get_minibatch(data_indices, feature_bucket, feature_length_bucket, sentence_bucket, batchsize, BLANK)
			x_batch = (x_batch - mean_x_batch) / stddev_x_batch

			if model.xp is cuda.cupy:
				x_batch = cuda.to_gpu(x_batch.astype(np.float32))
				t_batch = cuda.to_gpu(np.asarray(t_batch).astype(np.int32))
				x_length_batch = cuda.to_gpu(np.asarray(x_length_batch).astype(np.int32))
				t_length_batch = cuda.to_gpu(np.asarray(t_length_batch).astype(np.int32))

			y_batch = model(x_batch, split_into_variables=False)
			y_batch = xp.argmax(y_batch.data, axis=2)

			for argmax_sequence, true_sequence in zip(y_batch, t_batch):
				target_sequence = []
				for token in true_sequence:
					if token == BLANK:
						continue
					target_sequence.append(int(token))
				pred_seqence = []
				prev_token = BLANK
				for token in argmax_sequence:
					if token == BLANK:
						prev_token = BLANK
						continue
					if token == prev_token:
						continue
					pred_seqence.append(int(token))
					prev_token = token
				# if approximate == True:
				# 	print("true:", target_sequence, "pred:", pred_seqence)
				error = compute_character_error_rate(target_sequence, pred_seqence)
				sum_error += error


			sys.stdout.write("\r" + stdout.CLEAR)
			sys.stdout.write("\rComputing error - bucket {}/{} - iteration {}/{}".format(bucket_idx + 1, len(buckets_indices), itr, total_iterations))
			sys.stdout.flush()
			data_indices = np.roll(data_indices, batchsize)

		errors.append(sum_error * 100.0 / batchsize / total_iterations)
	return errors
