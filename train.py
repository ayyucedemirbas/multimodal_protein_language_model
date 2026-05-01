import tensorflow as tf
import importlib

import layers
import encoder
import decoder
import model
import dataset

importlib.reload(layers)
importlib.reload(encoder)
importlib.reload(decoder)
importlib.reload(model)
importlib.reload(dataset)

from dataset import load_tape_secondary_structure
from model import MultimodalProteinModel, CustomLearningRateScheduler

def main():
    datasets, vocabs = load_tape_secondary_structure(
        data_dir="./tape_data",
        max_length=512,
        batch_size=32,
        num_classes=3,
        splits=("train", "valid")
    )
    
    train_ds = datasets["train"]
    valid_ds = datasets["valid"]
    
    amino_acid_vocab_size = len(vocabs["aa"])
    structure_vocab_size = len(vocabs["ss"])
    
    print("Initializing the model...")
    d_model = 256
    protein_model = MultimodalProteinModel(
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_model=d_model,
        num_heads=8,
        d_ff=1024,
        num_experts=4,
        k=2,
        amino_acid_vocab_size=amino_acid_vocab_size,
        structure_vocab_size=structure_vocab_size,
        max_seq_length=512,
        dropout_rate=0.1,
        label_smoothing=0.1
    )
    
    learning_rate = CustomLearningRateScheduler(d_model=d_model, warmup_steps=4000)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    protein_model.compile(optimizer=optimizer)
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1
    )
    
    history = protein_model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=10,
        callbacks=[checkpoint_cb]
    )

if __name__ == "__main__":
    main()
