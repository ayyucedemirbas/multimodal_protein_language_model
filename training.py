import tensorflow as tf
import os
from model import MultimodalProteinModel, CustomLearningRateScheduler
from preprocessing import preprocess_protein_sequence, preprocess_structure_data

def train_multimodal_protein_model(
    protein_seqs,
    structure_data,
    structural_images=None,
    amino_acid_vocab=None,
    structure_vocab=None,
    batch_size=32,
    epochs=10,
    max_seq_length=1024,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    num_experts=8,
    k=2,
    dropout_rate=0.1,
    checkpoint_path="./checkpoints/protein_model"
):
    
    # Create vocabularies if not provided
    if amino_acid_vocab is None:
        # Standard amino acid vocabulary plus special tokens
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        amino_acid_vocab = {aa: i+3 for i, aa in enumerate(amino_acids)}
        amino_acid_vocab["<PAD>"] = 0
        amino_acid_vocab["<START>"] = 1
        amino_acid_vocab["<END>"] = 2
        amino_acid_vocab["<UNK>"] = 3
    
    if structure_vocab is None:
        structure_elements = ["H", "E", "C", "T", "S", "G", "B"]  # Example structure elements
        structure_vocab = {elem: i+3 for i, elem in enumerate(structure_elements)}
        structure_vocab["<PAD>"] = 0
        structure_vocab["<START>"] = 1
        structure_vocab["<END>"] = 2
        structure_vocab["<UNK>"] = 3
    
    amino_acid_vocab_size = len(amino_acid_vocab)
    structure_vocab_size = len(structure_vocab)
    
    processed_seqs = [preprocess_protein_sequence(seq, max_seq_length, amino_acid_vocab) 
                     for seq in protein_seqs]
    
    processed_structures = [preprocess_structure_data(struct, max_seq_length, structure_vocab) 
                           for struct in structure_data]
    
    if structural_images is not None:
        dataset = tf.data.Dataset.from_tensor_slices((processed_seqs, processed_structures, structural_images))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((processed_seqs, processed_structures))
    
    dataset = dataset.cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    model = MultimodalProteinModel(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_experts=num_experts,
        k=k,
        amino_acid_vocab_size=amino_acid_vocab_size,
        structure_vocab_size=structure_vocab_size,
        max_seq_length=max_seq_length,
        dropout_rate=dropout_rate
    )
    

    learning_rate = CustomLearningRateScheduler(d_model)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    
    history = model.fit(
        dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )
    
    return model, history, amino_acid_vocab, structure_vocab
