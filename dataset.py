"""
dataset.py – Download and prepare the TAPE secondary structure dataset.

Dataset
-------
TAPE (Tasks Assessing Protein Embeddings) secondary structure benchmark
  • Source  : https://github.com/songlab-cal/tape
  • Download: ~150 MB tarball hosted on AWS S3 (public, no login required)
  • Labels  : 3-class (H / E / C) *and* 8-class (DSSP) per-residue labels
  • Splits  : train (~8 500 chains), valid (~2 100 chains), test sets
              CB513 / TS115 / CASP12

Usage
-----
    from dataset import load_tape_secondary_structure

    train_ds, valid_ds, vocabs = load_tape_secondary_structure(
        data_dir="./tape_data",
        max_length=512,
        batch_size=32,
        num_classes=3,     # 3 or 8
    )
    # train_ds yields (seq_tokens, struct_tokens) batches ready for the model.
"""

import os
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing import (
    build_amino_acid_vocab,
    build_structure_vocab,
    SS3_ELEMENTS,
    SS8_ELEMENTS,
    preprocess_protein_sequence,
    preprocess_structure_data,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TAPE_SS_URL = (
    "https://s3.amazonaws.com/songlabdata/proteindata/"
    "data_raw_pytorch/secondary_structure.tar.gz"
)

SPLIT_FILES = {
    "train":  "secondary_structure/secondary_structure_train.json",
    "valid":  "secondary_structure/secondary_structure_valid.json",
    "cb513":  "secondary_structure/secondary_structure_cb513.json",
    "ts115":  "secondary_structure/secondary_structure_ts115.json",
    "casp12": "secondary_structure/secondary_structure_casp12.json",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download_tape(data_dir: str) -> None:
    """Download and extract the TAPE secondary-structure tarball if needed."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tarball = data_dir / "secondary_structure.tar.gz"
    marker  = data_dir / "secondary_structure" / ".done"

    if marker.exists():
        print("[dataset] TAPE data already downloaded.")
        return

    print(f"[dataset] Downloading TAPE secondary structure dataset to {data_dir} …")
    print(f"          URL: {TAPE_SS_URL}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = downloaded / total_size * 100 if total_size > 0 else 0
        print(f"\r  {pct:5.1f}%  ({downloaded // 1_048_576} / {total_size // 1_048_576} MB)",
              end="", flush=True)

    urllib.request.urlretrieve(TAPE_SS_URL, tarball, reporthook=_progress)
    print()

    print("[dataset] Extracting …")
    with tarfile.open(tarball, "r:gz") as tf_tar:
        tf_tar.extractall(data_dir)

    marker.touch()
    print("[dataset] Done.")


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------
def _load_split(data_dir: str, split: str):
    """Load a TAPE JSON split and return (sequences, ss3_labels, ss8_labels)."""
    path = Path(data_dir) / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            "Did you forget to call _download_tape(data_dir)?"
        )

    with open(path) as fh:
        records = json.load(fh)

    sequences, ss3_list, ss8_list = [], [], []
    for rec in records:
        seq  = rec["primary"]                   # amino-acid string
        ss3  = "".join(str(s) for s in rec["ss3"])
        ss8  = "".join(str(s) for s in rec["ss8"])

        # Map integer labels back to letters if needed
        ss3 = _int_to_label(ss3, SS3_ELEMENTS)
        ss8 = _int_to_label(ss8, SS8_ELEMENTS)

        sequences.append(seq)
        ss3_list.append(ss3)
        ss8_list.append(ss8)

    return sequences, ss3_list, ss8_list


def _int_to_label(label_str: str, elements: list) -> str:
    """Convert a string of digit characters to letter labels, if needed."""
    if label_str and label_str[0].isdigit():
        return "".join(elements[int(c)] for c in label_str)
    return label_str


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------
def _make_tf_dataset(
    sequences,
    structures,
    aa_vocab,
    ss_vocab,
    max_length,
    batch_size,
    shuffle=False,
):
    def _gen():
        for seq, ss in zip(sequences, structures):
            seq_t = preprocess_protein_sequence(seq, max_length, aa_vocab)
            ss_t  = preprocess_structure_data(ss,  max_length, ss_vocab)
            yield seq_t, ss_t

    output_sig = (
        tf.TensorSpec(shape=(max_length,),     dtype=tf.int32),
        tf.TensorSpec(shape=(max_length + 2,), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(_gen, output_signature=output_sig)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(sequences), 5_000), seed=42)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_tape_secondary_structure(
    data_dir: str = "./tape_data",
    max_length: int = 512,
    batch_size: int = 32,
    num_classes: int = 3,
    splits: tuple = ("train", "valid", "cb513"),
    download: bool = True,
):
    """Download (once) and return ``tf.data.Dataset`` objects for TAPE SS.

    Parameters
    ----------
    data_dir    : local directory for the dataset files
    max_length  : sequences longer than this are truncated
    batch_size  : batch size for all returned datasets
    num_classes : 3 (H/E/C coarse labels) or 8 (full DSSP labels)
    splits      : which splits to return (subset of train/valid/cb513/ts115/casp12)
    download    : set False to skip the download step (data must exist already)

    Returns
    -------
    datasets : dict  {split_name: tf.data.Dataset}
    vocabs   : dict  {"aa": amino_acid_vocab, "ss": structure_vocab}
    """
    if download:
        _download_tape(data_dir)

    elements = SS3_ELEMENTS if num_classes == 3 else SS8_ELEMENTS
    aa_vocab = build_amino_acid_vocab()
    ss_vocab = build_structure_vocab(elements)

    datasets = {}
    for split in splits:
        print(f"[dataset] Loading split '{split}' …")
        seqs, ss3, ss8 = _load_split(data_dir, split)
        labels = ss3 if num_classes == 3 else ss8
        ds = _make_tf_dataset(
            seqs, labels, aa_vocab, ss_vocab,
            max_length=max_length,
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
        datasets[split] = ds
        print(f"           {len(seqs):,} sequences loaded.")

    vocabs = {"aa": aa_vocab, "ss": ss_vocab}
    return datasets, vocabs


# ---------------------------------------------------------------------------
# Quick stats helper
# ---------------------------------------------------------------------------
def dataset_stats(data_dir: str = "./tape_data", split: str = "train"):
    """Print basic statistics for one split (useful for sanity-checking)."""
    _download_tape(data_dir)
    seqs, ss3, _ = _load_split(data_dir, split)
    lengths = [len(s) for s in seqs]
    print(f"Split : {split}")
    print(f"Chains: {len(seqs):,}")
    print(f"Len   : min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")
    label_counts = {}
    for s in ss3:
        for c in s:
            label_counts[c] = label_counts.get(c, 0) + 1
    total = sum(label_counts.values())
    print("SS3 class distribution:")
    for k in sorted(label_counts):
        print(f"  {k}: {label_counts[k]:,}  ({label_counts[k]/total*100:.1f}%)")


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./tape_data")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        dataset_stats(args.data_dir)
    else:
        datasets, vocabs = load_tape_secondary_structure(data_dir=args.data_dir)
        print("Vocab sizes — AA:", len(vocabs["aa"]), "| SS:", len(vocabs["ss"]))
        for split, ds in datasets.items():
            print(f"Split '{split}': {ds.cardinality()} batches")