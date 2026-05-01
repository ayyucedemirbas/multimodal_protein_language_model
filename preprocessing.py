import tensorflow as tf


STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

def build_amino_acid_vocab():
    vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    for i, aa in enumerate(STANDARD_AA):
        vocab[aa] = i + 4
    vocab["X"] = 3   # unknown residue → <UNK>
    return vocab


def build_structure_vocab(elements=None):
    if elements is None:
        elements = list("HECSTGBI")   # 8-class DSSP labels
    vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    for i, el in enumerate(elements):
        vocab[el] = i + 4
    return vocab


# Convenient 3-class mapping used by TAPE/CB513 benchmarks
SS3_ELEMENTS = ["H", "E", "C"]   # helix / strand / coil
SS8_ELEMENTS = list("HECSTGBI")


def preprocess_protein_sequence(sequence: str, max_length: int, vocab: dict):
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
    tokens = [vocab.get("<START>", 1)]
    tokens += [vocab.get(el, vocab.get("<UNK>", 3)) for el in structure_data]
    tokens += [vocab.get("<END>", 2)]

    target_len = max_length + 2   # +2 for START / END

    if len(tokens) > target_len:
        tokens = tokens[: target_len - 1] + [vocab.get("<END>", 2)]
    else:
        tokens = tokens + [vocab.get("<PAD>", 0)] * (target_len - len(tokens))

    return tf.convert_to_tensor(tokens, dtype=tf.int32)


def batch_encode_sequences(sequences, max_length: int, vocab: dict):
    tensors = [preprocess_protein_sequence(s, max_length, vocab) for s in sequences]
    return tf.stack(tensors)   # (N, max_length)


def batch_encode_structures(structures, max_length: int, vocab: dict):
    tensors = [preprocess_structure_data(s, max_length, vocab) for s in structures]
    return tf.stack(tensors)   # (N, max_length + 2)
