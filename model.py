import tensorflow as tf
from encoder import ProteinEncoder
from decoder import ProteinDecoder


class MultimodalProteinModel(tf.keras.Model):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        d_model,
        num_heads,
        d_ff,
        num_experts,
        k,
        amino_acid_vocab_size,
        structure_vocab_size,
        max_seq_length,
        dropout_rate=0.1,
        label_smoothing=0.1,
    ):
        super().__init__()

        self.encoder = ProteinEncoder(
            num_encoder_layers, d_model, num_heads, d_ff,
            num_experts, k, amino_acid_vocab_size, max_seq_length, dropout_rate,
        )

        self.decoder = ProteinDecoder(
            num_decoder_layers, d_model, num_heads, d_ff,
            num_experts, k, structure_vocab_size, max_seq_length + 2, dropout_rate,
        )

        self.image_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64,  (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(d_model),
        ])

        self.fusion_layer = tf.keras.layers.Dense(d_model)
        self.final_layer  = tf.keras.layers.Dense(structure_vocab_size)

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=label_smoothing,
            reduction="none",
        )

        self.loss_tracker     = tf.keras.metrics.Mean(name="loss")
        self.accuracy_metric  = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_acc_tracker  = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

        self.structure_vocab_size = structure_vocab_size

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    def create_masks(self, inp, tar):
        enc_padding_mask  = self.create_padding_mask(inp)
        dec_padding_mask  = self.create_padding_mask(inp)
        look_ahead_mask   = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_mask   = self.create_padding_mask(tar)
        combined_mask     = tf.maximum(dec_target_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask

    def call(self, protein_seq, structure_targets, structural_image=None, training=None):
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
            protein_seq, structure_targets
        )

        enc_output = self.encoder(protein_seq, training=training, mask=enc_padding_mask)

        if structural_image is not None:
            image_features = self.image_encoder(structural_image, training=training)
            image_features = tf.expand_dims(image_features, axis=1)
            seq_len        = tf.shape(enc_output)[1]
            image_features = tf.repeat(image_features, seq_len, axis=1)
            enc_output     = self.fusion_layer(
                tf.concat([enc_output, image_features], axis=-1)
            )

        dec_output, attention_weights = self.decoder(
            structure_targets, 
            enc_output, 
            training=training, 
            look_ahead_mask=combined_mask, 
            padding_mask=dec_padding_mask
        )
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


    def _masked_loss(self, real, pred):
        real_oh = tf.one_hot(real, self.structure_vocab_size)  # for label-smoothing loss
        per_token_loss = self.loss_fn(real_oh, pred)            # (B, T)
        mask = tf.cast(tf.not_equal(real, 0), tf.float32)
        loss = tf.reduce_sum(per_token_loss * mask) / (tf.reduce_sum(mask) + 1e-8)
        return loss

    def train_step(self, data):
        if len(data) == 3:
            protein_seq, structure_targets, structural_image = data
        else:
            protein_seq, structure_targets = data
            structural_image = None

        dec_input  = structure_targets[:, :-1]
        real_output = structure_targets[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self(
                protein_seq, 
                structure_targets=dec_input, 
                structural_image=structural_image, 
                training=True
            )
            loss = self._masked_loss(real_output, predictions)

        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.accuracy_metric.update_state(real_output, predictions)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_metric.result()}

    def test_step(self, data):
        if len(data) == 3:
            protein_seq, structure_targets, structural_image = data
        else:
            protein_seq, structure_targets = data
            structural_image = None

        dec_input   = structure_targets[:, :-1]
        real_output = structure_targets[:, 1:]

        predictions, _ = self(
            protein_seq, 
            structure_targets=dec_input, 
            structural_image=structural_image, 
            training=False
        )
        loss = self._masked_loss(real_output, predictions)

        self.val_loss_tracker.update_state(loss)
        self.val_acc_tracker.update_state(real_output, predictions)
        return {
            "val_loss": self.val_loss_tracker.result(),
            "val_accuracy": self.val_acc_tracker.result(),
        }

    @property
    def metrics(self):
        return [
            self.loss_tracker, self.accuracy_metric,
            self.val_loss_tracker, self.val_acc_tracker,
        ]


class CustomLearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Transformer warm-up schedule: lr ∝ d_model^-0.5 · min(step^-0.5, step·warmup^-1.5)."""

    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step  = tf.cast(step, tf.float32)
        arg1  = tf.math.rsqrt(step)
        arg2  = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": int(self.d_model.numpy()), "warmup_steps": self.warmup_steps}
