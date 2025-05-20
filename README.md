**Multimodal Protein Language Model**

This documentation provides an overview, installation instructions, usage examples, and API reference for the `multimodal_protein_language_model` repository by ayyucedemirbas. It supports sequence-to-structure/function prediction using transformer-based encoder-decoder architecture with a mixture-of-experts and optional structural image input.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)

   * [Encoder](#proteinencoder)
   * [Decoder](#proteindecoder)
   * [MultimodalFusion](#multimodalfusion)
   * [Custom Learning Rate](#custom-learning-rate)
6. [Training](#training)
7. [Usage Examples](#usage-examples)
8. [API Reference](#api-reference)
9. [License](#license)

---

## Overview

The `MultimodalProteinModel` integrates:

* **Protein Sequence Encoder** based on transformer layers with mixture-of-experts routing.
* **Protein Structure/Function Decoder** generating structural tokens.
* **Image Encoder** for optional 2D structural data to perform multimodal fusion.
* **Custom learning rate scheduler** following the "Attention Is All You Need" warmup strategy.

Use cases include predicting protein secondary/tertiary structures, binding sites, or functional motifs, optionally guided by structural images.

---

## Repository Structure

```text
multimodal_protein_language_model/
├── README.md            # Minimal original readme
├── LICENSE              # MIT License
├── encoder.py           # Transformer encoder with MoE layers
├── decoder.py           # Transformer decoder with MoE layers
├── layers.py            # Core MultiheadAttention, MixtureOfExperts, positional encoding
├── model.py             # Complete MultimodalProteinModel class
├── preprocessing.py     # Sequence and structure tokenization utilities
└── training.py          # High-level training routine and entry point
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ayyucedemirbas/multimodal_protein_language_model.git
   cd multimodal_protein_language_model
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install tensorflow numpy
   ```

---

## Data Preprocessing

Two helper functions in `preprocessing.py`:

* `preprocess_protein_sequence(sequence: str, max_length: int, vocab: dict) -> tf.Tensor`
  Converts an amino acid sequence to integer tokens, pads/truncates to `max_length`.

* `preprocess_structure_data(structure_data: List[str], max_length: int, vocab: dict) -> tf.Tensor`
  Converts structure tokens (e.g., secondary structure labels) to integers, adds start/end tokens, pads/truncates.

**Example**:

```python
from preprocessing import preprocess_protein_sequence, preprocess_structure_data
# Sample vocab
aa_vocab = {aa: i+3 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
aa_vocab.update({"<PAD>":0, "<START>":1, "<END>":2, "<UNK>":3})
seq_tensor = preprocess_protein_sequence("ACDIPK", max_length=10, vocab=aa_vocab)
```

---

## Model Architecture

### `ProteinEncoder` (<u>encoder.py</u>)

* **Layers**: Embedding, positional encoding, `num_layers` of `EncoderLayer`.
* **EncoderLayer**: Multi-head self-attention (with dropout & layer norm) + Mixture-of-Experts feed-forward.

```python
from encoder import ProteinEncoder
encoder = ProteinEncoder(
    num_layers=6, d_model=512, num_heads=8,
    d_ff=2048, num_experts=8, k=2,
    amino_acid_vocab_size=24, max_position=1024,
    dropout_rate=0.1
)
enc_output = encoder(input_seq_tensor)
```

### `ProteinDecoder` (<u>decoder.py</u>)

* **Layers**: Embedding, positional encoding, `num_layers` of `DecoderLayer`.
* **DecoderLayer**: Masked self-attention + encoder-decoder cross-attention + MoE feed-forward.

```python
from decoder import ProteinDecoder
decoder = ProteinDecoder(
    num_layers=6, d_model=512, num_heads=8,
    d_ff=2048, num_experts=8, k=2,
    target_vocab_size=structure_vocab_size,
    max_position=1024
)
logits, attn_weights = decoder(target_tokens, enc_output)
```

### Multimodal Fusion (<u>model.py</u>)

1. **Image Encoder**: 3 Conv2D + MaxPool blocks, Flatten, Dense to `d_model`.
2. **Fusion**: Concatenate sequence features and repeated image features, project via `Dense(d_model)`.

### Custom Learning Rate Scheduler

```python
from model import CustomLearningRateScheduler
lr_schedule = CustomLearningRateScheduler(d_model=512, warmup_steps=4000)
optimizer = tf.keras.optimizers.Adam(lr_schedule)
```

---

## Training (`training.py`)

`train_multimodal_protein_model(...)` orchestrates preprocessing, dataset creation, model compilation, and training.

### Key Arguments:

* `protein_seqs`: List of strings (amino acid sequences).
* `structure_data`: List of lists/strings of structure labels.
* `structural_images`: Optional array of image tensors.
* `batch_size`, `epochs`, model hyperparameters, `checkpoint_path`.

**Example Usage**:

```python
from training import train_multimodal_protein_model
# Dummy data
protein_seqs = ["ACDEFGHIKLMNPQRS"]
structure_data = [["H","E","C","C"]]
# Train
model, history, aa_vocab, struct_vocab = train_multimodal_protein_model(
    protein_seqs, structure_data, epochs=5, batch_size=2
)
```

---

## API Reference

### `layers.py`

* **MultiheadAttention**: `call([q,k,v], mask=None, training=None)` → `(output, attn_weights)`
* **ExpertLayer**: Feed-forward sub-layer.
* **MixtureOfExperts**: `call(x, training=None)` → gated MoE output.
* \*\*positional\_encoding(position, d\_model)`→ Tensor of shape`(1, position, d\_model)\`

### `model.py`

* **MultimodalProteinModel**:

  * `call((protein_seq, structure_targets, structural_image), training)` → `(logits, attention_weights)`
  * `train_step(data)` → dict with `'loss'` and `'accuracy'`
  * `.create_masks(inp, tar)` → `(enc_padding_mask, combined_mask, dec_padding_mask)`
  * `.metrics` property → `[loss_tracker, accuracy_metric]`

### `training.py`

* \*\*train\_multimodal\_protein\_model(...)`** → `(model, history, amino\_acid\_vocab, structure\_vocab)\`

---

## License

This project is licensed under the [GNU GENERAL PUBLIC LICENSE Version 3](/LICENSE). Feel free to use and modify.

---

