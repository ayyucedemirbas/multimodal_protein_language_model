import tensorflow as tf
import numpy as np


class MultiheadAttention(tf.keras.layers.Layer):
    """Multi-head self/cross attention.

    Bug fixed: the original code computed the residual as
    ``output + q[:, :, 0, :]``, which silently picked a single head-slice of
    the *projected* query tensor instead of the original input.  The fix
    captures ``inputs[0]`` before any projection.
    """

    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        # FIX: Added this to prevent Keras warnings about destroyed mask info
        self.supports_masking = True
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # ------------------------------------------------------------------
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # ------------------------------------------------------------------
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled += mask * -1e9

        attention_weights = tf.nn.softmax(scaled, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    # ------------------------------------------------------------------
    def call(self, inputs, mask=None, training=None):
        q_input, k_input, v_input = inputs   # keep original query for residual
        batch_size = tf.shape(q_input)[0]

        q = self.split_heads(self.wq(q_input), batch_size)
        k = self.split_heads(self.wk(k_input), batch_size)
        v = self.split_heads(self.wv(v_input), batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # Merge heads back
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat)
        output = self.dropout(output, training=training)

        # FIX: residual uses the *original* query input, not a head-slice
        output = self.layernorm(output + q_input)

        return output, attention_weights


# ---------------------------------------------------------------------------
class ExpertLayer(tf.keras.layers.Layer):
    """Single FFN expert used inside MixtureOfExperts."""

    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(d_ff, activation="gelu")
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dropout(out, training=training)
        return self.layernorm(x + out)


# ---------------------------------------------------------------------------
class MixtureOfExperts(tf.keras.layers.Layer):
    """Sparse Top-K Mixture-of-Experts FFN.

    Bug fixed: the original code tried to scatter expert outputs back via
    ``tf.pad(..., [[0, N - n], [0, 0]])`` which assumes selected tokens
    sit at position 0 — giving silently wrong results for all other tokens.
    The fix uses ``tf.tensor_scatter_nd_add`` to write each expert's output
    to the correct positions.
    """

    def __init__(self, d_model, d_ff, num_experts, k=2, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k

        self.experts = [ExpertLayer(d_model, d_ff, dropout_rate) for _ in range(num_experts)]
        self.router = tf.keras.layers.Dense(num_experts, use_bias=False)

    # ------------------------------------------------------------------
    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        x_flat = tf.reshape(x, [-1, self.d_model])          # [B*S, d_model]

        router_logits = self.router(x_flat)                  # [B*S, num_experts]
        expert_gates, expert_indices = tf.math.top_k(router_logits, k=self.k)
        expert_gates = tf.nn.softmax(expert_gates, axis=-1)  # [B*S, k]

        final_output = tf.zeros_like(x_flat)                 # [B*S, d_model]

        for i, expert in enumerate(self.experts):
            # Boolean mask: which (token, slot) pairs route to expert i
            is_this_expert = tf.equal(expert_indices, i)     # [B*S, k]

            # Aggregate gate weight from all k slots for this expert
            gate_i = tf.reduce_sum(
                expert_gates * tf.cast(is_this_expert, tf.float32), axis=-1
            )                                                 # [B*S]

            # FIX: Used reshape instead of squeeze, as it is safer for dynamic shapes
            token_indices = tf.reshape(tf.where(gate_i > 0), [-1])

            # FIX: Removed the non-graph-friendly 'if size == 0' check entirely.
            # Empty tensor inputs flow seamlessly through tf.gather and dense layers.

            selected_inputs = tf.gather(x_flat, token_indices)   # [sel, d_model]
            expert_out = expert(selected_inputs, training=training)

            # Scale by gate values
            selected_gates = tf.expand_dims(
                tf.gather(gate_i, token_indices), axis=-1
            )                                                  # [sel, 1]
            weighted_out = expert_out * selected_gates         # [sel, d_model]

            # FIX: scatter to the correct token positions
            final_output = tf.tensor_scatter_nd_add(
                final_output,
                tf.expand_dims(token_indices, axis=1),         # [[idx], ...]
                weighted_out,
            )

        return tf.reshape(final_output, [batch_size, seq_len, self.d_model])


# ---------------------------------------------------------------------------
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)