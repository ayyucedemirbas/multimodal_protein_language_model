import tensorflow as tf


# ---------------------------------------------------------------------------
# Default vocabularies
# ---------------------------------------------------------------------------
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

def build_amino_acid_vocab():
    """Standard 20 amino acids + special tokens + 'X' for unknowns."""
    vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    for i, aa in enumerate(STANDARD_AA):
        vocab[aa] = i + 4
    vocab["X"] = 3   # unknown residue → <UNK>
    return vocab


def build_structure_vocab(elements=None):
    """3-class (H/E/C) or 8-class (HECSGTBI) secondary structure vocabulary."""
    if elements is None:
        elements = list("HECSTGBI")   # 8-class DSSP labels
    vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    for i, el in enumerate(elements):
        vocab[el] = i + 4
    return vocab


# Convenient 3-class mapping used by TAPE/CB513 benchmarks
SS3_ELEMENTS = ["H", "E", "C"]   # helix / strand / coil
SS8_ELEMENTS = list("HECSTGBI")


# ---------------------------------------------------------------------------
# Preprocessing functions
# ---------------------------------------------------------------------------
def preprocess_protein_sequence(sequence: str, max_length: int, vocab: dict):
    """Tokenise an amino-acid string and pad / truncate to ``max_length``.

    Improvement vs. original: the sequence is now wrapped with START/END tokens
    (matching how structure targets are already encoded) so the encoder sees
    explicit boundary signals.
    """
    tokens = [vocab.get("<START>", 1)]
    tokens += [vocab.get(aa, vocab.get("<UNK>", 3)) for aa in sequence.upper()]
    tokens += [vocab.get("<END>", 2)]

    if len(tokens) > max_length:
        # Keep START, truncate body, re-attach END
        tokens = tokens[: max_length - 1] + [vocab.get("<END>", 2)]
    else:
        tokens = tokens + [vocab.get("<PAD>", 0)] * (max_length - len(tokens))

    return tf.convert_to_tensor(tokens, dtype=tf.int32)


def preprocess_structure_data(structure_data, max_length: int, vocab: dict):
    """Tokenise secondary-structure labels and pad / truncate.

    ``structure_data`` can be a string (e.g. ``"HHECCC"``) or a list of
    single-character labels.
    """
    tokens = [vocab.get("<START>", 1)]
    tokens += [vocab.get(el, vocab.get("<UNK>", 3)) for el in structure_data]
    tokens += [vocab.get("<END>", 2)]

    target_len = max_length + 2   # +2 for START / END

    if len(tokens) > target_len:
        tokens = tokens[: target_len - 1] + [vocab.get("<END>", 2)]
    else:
        tokens = tokens + [vocab.get("<PAD>", 0)] * (target_len - len(tokens))

    return tf.convert_to_tensor(tokens, dtype=tf.int32)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------
def batch_encode_sequences(sequences, max_length: int, vocab: dict):
    tensors = [preprocess_protein_sequence(s, max_length, vocab) for s in sequences]
    return tf.stack(tensors)   # (N, max_length)


def batch_encode_structures(structures, max_length: int, vocab: dict):
    tensors = [preprocess_structure_data(s, max_length, vocab) for s in structures]
    return tf.stack(tensors)   # (N, max_length + 2)