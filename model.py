import tensorflow as tf
from encoder import ProteinEncoder
from decoder import ProteinDecoder

class MultimodalProteinModel(tf.keras.Model):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, d_ff,
                 num_experts, k, amino_acid_vocab_size, structure_vocab_size, 
                 max_seq_length, dropout_rate=0.1):
        super(MultimodalProteinModel, self).__init__()
        
        # Main Encoder-Decoder for Protein Sequence to Structure/Function
        self.encoder = ProteinEncoder(num_encoder_layers, d_model, num_heads, d_ff,
                                     num_experts, k, amino_acid_vocab_size,
                                     max_seq_length, dropout_rate)
        
        self.decoder = ProteinDecoder(num_decoder_layers, d_model, num_heads, d_ff,
                                     num_experts, k, structure_vocab_size,
                                     max_seq_length, dropout_rate)
        
        # Image Encoder for multimodal processing (handling structural data)
        self.image_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(d_model)
        ])
        
        # Cross-modal fusion module
        self.fusion_layer = tf.keras.layers.Dense(d_model)
        
        # Final prediction layers
        self.final_layer = tf.keras.layers.Dense(structure_vocab_size)
        
        # Loss tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        
    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        
        # add extra dimensions to add the padding to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
    
    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)
        
        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)
        
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return enc_padding_mask, combined_mask, dec_padding_mask
    
    def call(self, inputs, training=None):
        # Unpack inputs
        protein_seq, structure_targets, structural_image = inputs
        
        # Create masks for the transformer
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(
            protein_seq, structure_targets)
        
        # Process sequence through encoder
        enc_output = self.encoder(protein_seq, training, enc_padding_mask)
        
        # Process structural image if available
        if structural_image is not None:
            image_features = self.image_encoder(structural_image, training=training)
            image_features = tf.expand_dims(image_features, axis=1)
            
            # Repeat image features to match sequence length
            seq_len = tf.shape(enc_output)[1]
            image_features = tf.repeat(image_features, seq_len, axis=1)
            
            # Fusion of sequence and image features
            enc_output = self.fusion_layer(tf.concat([enc_output, image_features], axis=-1))
        
        # Process through decoder
        dec_output, attention_weights = self.decoder(
            structure_targets, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        # Final prediction layer
        final_output = self.final_layer(dec_output)
        
        return final_output, attention_weights
    
    def train_step(self, data):
        # Unpack the data
        if len(data) == 3:
            protein_seq, structure_targets, structural_image = data
        else:
            protein_seq, structure_targets = data
            structural_image = None
        
        # Shift targets for teacher forcing (decoder input and real output)
        decoder_input = structure_targets[:, :-1]
        real_output = structure_targets[:, 1:]
        
        with tf.GradientTape() as tape:
            # Forward pass
            if structural_image is not None:
                predictions, _ = self((protein_seq, decoder_input, structural_image), training=True)
            else:
                predictions, _ = self((protein_seq, decoder_input, None), training=True)
            
            # Calculate loss
            loss = self.compiled_loss(real_output, predictions)
        
        # Calculate gradients and apply via optimizer
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_metric.update_state(real_output, predictions)
        
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_metric.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_metric]

# Custom learning rate scheduler with warmup
class CustomLearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomLearningRateScheduler, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
