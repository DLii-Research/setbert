import numpy as np
import os
import re
import shelve
import sys

import bootstrap

# A regular expression for recognizing the components of the fastq files
FASTQ_REGEX = r"(?<=0000_AG_)(\d{4})-(\d{2})-(\d{2})(?=.fastq)"

# Map DNA base calls to integers
BASE_MAP = {b: i for i, b in enumerate('ACGTN')}


def define_arguments(parser):
	parser.add_argument("--val-split", type=float, default=0.1)
	parser.add_argument("--test-split", type=float, default=0.1)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--shuffle", type=bool, default=True)


def encode_base(c):
    return BASE_MAP[c]


def parse_sequence(sequence):
	return bytes(list(map(encode_base, sequence.rstrip())))


def process_sample(inpath, outpath, filename, val_split, test_split, shuffle):
	"""
	Process a sample by extracting all DNA sequences from the given fastq file
	"""
	with open(inpath) as f:
		print(f"{inpath} -> {outpath}")
		sequences = f.readlines()[1::4]
		n = len(sequences)

		n_val = int(n*(val_split))
		n_test = int(n*(test_split))
		n_train = n - n_val - n_test
		ends = np.cumsum((n_train, n_val, n_test))
		folders = ("train", "validation", "test")

		start = 0
		if shuffle:
			indices = np.random.permutation(n)
		else:
			indices = np.arange(n)
		for end, folder in zip(ends, folders):
			if start == end:
				continue
			store = shelve.open(os.path.join(outpath, folder, filename))
			for key, i in enumerate(range(start, end)):
				sequence = sequences[indices[i]]
				store[str(key)] = parse_sequence(sequence)
			store.close()
			start = end


def process_season(path, season, val_split, test_split, shuffle, writedir):
	"""
	Process a season-folder of samples
	"""
	for file in os.listdir(path):
		if not file.endswith(".fastq"):
			continue
		date = re.search(FASTQ_REGEX, file)[0]
		inpath = os.path.join(path, file)
		filename = f"{season.lower()}_{date}"
		process_sample(inpath, writedir, filename, val_split, test_split, shuffle)


def create_dataset(config):
	"""
	Process a DNA sample dataset
	"""
	if not os.path.isdir(config.data_path):
		raise Exception(f"Invalid input directory: {config.data_path}")

	write_path = bootstrap.data_path("dataset")

	for folder in ("train", "validation", "test"):
		os.makedirs(os.path.join(write_path, folder), exist_ok=True)

	for season in os.listdir(config.data_path):
		path = os.path.join(config.data_path, season)
		if not os.path.isdir(path):
			continue
		process_season(
			path,
			season,
			config.val_split,
			config.test_split,
			config.shuffle,
			write_path)


def main(argv):
	# Job Information
	job_info = {
		"name": "dataset-dnasamples",
		"job_type": bootstrap.JobType.CreateDataset,
		"group": "dataset/dnasamples"
	}

	_, config = bootstrap.init(argv, job_info, define_arguments)

	# Set the seed if supplied
	if config.seed is not None:
		np.random.seed(config.seed)

	# Create the datasets
	create_dataset(config)

	# Log the dataset as an artifact
	bootstrap.log_dataset_artifact("dnasamples", paths="dataset", metadata={
		"train_split": 1.0 - config.val_split - config.test_split,
		"val_split": config.val_split,
		"test_split": config.test_split
	})


if __name__ == "__main__":
	sys.exit(main(sys.argv) or 0)
