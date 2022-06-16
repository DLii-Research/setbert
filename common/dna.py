import numpy as np
from . utils import static_vars

BASES = "ACGTN"

# General DNA Utilities ----------------------------------------------------------------------------

@static_vars(m = {c: i for i, c in enumerate(BASES)})
def encode_sequence(sequence: str):
    """
    Encode a DNA sequence into an integer vector representation.
    """
    return np.array([encode_sequence.m[base] for base in sequence])


@static_vars(m = [c for c in BASES])
def decode_sequence(sequence: list):
    """
    Decode a DNA sequence integer vector representation into a string of bases.
    """
    return ''.join(decode_sequence.m[base] for base in sequence)


def encode_kmers(encoded_sequence: list, kmer):
    """
    Encode a sequence vector representation as kmers.
    """
    return np.convolve(encoded_sequence, 5**np.arange(kmer), mode="valid")


def decode_kmers(kmer_sequence, kmer):
    """
    Decode a kmer sequence vector representation.
    """
    seq_len = len(kmer_sequence) + kmer - 1
    result = np.empty(shape=(seq_len), dtype=int)
    for i, j in enumerate(range(kmer - 1, -1, -1)):
        result[i:seq_len//kmer*kmer:kmer] = (kmer_sequence[::kmer] // (5**j)) % 5
    # Handle partial ending kmer
    for i, j in enumerate(range(seq_len % kmer - 1, -1, -1)):
        result[seq_len - (seq_len % kmer) + i] = (kmer_sequence[-1] // (5**j)) % 5
    return result

# Quality Score Utilities --------------------------------------------------------------------------

def encode_phred(probs, encoding=33):
    scores = (-10 * np.log10(np.array(probs))).astype(int)
    return ''.join((chr(score + encoding)) for score in scores)


def decode_phred(quality_str, encoding=33):
    scores = np.array([(ord(token) - encoding) for token in quality_str])
    return 10**(scores / -10)


