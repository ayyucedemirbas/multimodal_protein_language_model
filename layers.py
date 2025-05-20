import tensorflow as tf
import numpy as np

class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, mask=None, training=None):
        q, k, v = inputs
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        output = self.dropout(output, training=training)
        output = self.layernorm(output + q[:, :, 0, :])
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  
        
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        
        return output, attention_weights

class ExpertLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(ExpertLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.dense1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training=None):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dropout(out, training=training)
        return self.layernorm(x + out)

class MixtureOfExperts(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, num_experts, k=2, dropout_rate=0.1):
        super(MixtureOfExperts, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k  # Top-k gating: select k experts per token
        
        self.experts = [ExpertLayer(d_model, d_ff, dropout_rate) for _ in range(num_experts)]
        
        # Router (gate) to select experts
        self.router = tf.keras.layers.Dense(num_experts, use_bias=False)
        
    def call(self, x, training=None):
        batch_size, seq_len, d_model = tf.shape(x)[0], tf.shape(x)[1], self.d_model
        x_flat = tf.reshape(x, [-1, d_model])  # [batch_size * seq_len, d_model]
        
        # Get router logits and probabilities
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        
        # Get top-k experts per token
        expert_gate, expert_index = tf.math.top_k(router_logits, k=self.k)
        
        # Normalize the gate values within the top-k
        expert_gate = tf.nn.softmax(expert_gate, axis=-1)
        
        # One-hot encode the expert indices
        expert_mask = tf.one_hot(expert_index, depth=self.num_experts)  # [batch_size * seq_len, k, num_experts]
        
        # Expand and reshape inputs for parallel expert computation
        x_expanded = tf.expand_dims(x_flat, axis=1)  # [batch_size * seq_len, 1, d_model]
        x_expanded = tf.tile(x_expanded, [1, self.k, 1])  # [batch_size * seq_len, k, d_model]
        
        # Initialize final output
        final_output = tf.zeros_like(x_flat)
        
        # Process each expert 
        for i, expert in enumerate(self.experts):
            # Extract which positions need to be processed by this expert
            expert_mask_i = expert_mask[:, :, i]  # [batch_size * seq_len, k]
            expert_gate_i = expert_gate * expert_mask_i  # [batch_size * seq_len, k]
            
            # Expert selection: when expert_mask_i is 1, process with current expert
            # Otherwise, use zeros
            positions = tf.where(tf.reduce_sum(expert_mask_i, axis=1) > 0)
            if tf.shape(positions)[0] > 0:
                selected_inputs = tf.gather(x_flat, positions[:, 0])
                expert_outputs = expert(selected_inputs, training=training)
                
                # Compute weighted expert outputs by gates and add to final output
                for j in range(self.k):
                    gate_values = tf.expand_dims(expert_gate_i[:, j], axis=1)
                    weighted_output = gate_values * tf.pad(
                        expert_outputs,
                        [[0, tf.shape(x_flat)[0] - tf.shape(expert_outputs)[0]], [0, 0]],
                        mode='CONSTANT')
                    final_output += weighted_output
                    
        # Reshape back to original dimensions
        return tf.reshape(final_output, [batch_size, seq_len, d_model])

# Positional Encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)
