import tensorflow as tf
from layers import MultiheadAttention, MixtureOfExperts, positional_encoding

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, num_experts, k=2, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiheadAttention(d_model, num_heads, dropout_rate)
        self.moe = MixtureOfExperts(d_model, d_ff, num_experts, k, dropout_rate)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=None, mask=None):
        attn_output, _ = self.mha([inputs, inputs, inputs], mask, training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        moe_output = self.moe(out1, training=training)
        moe_output = self.dropout2(moe_output, training=training)
        out2 = self.layernorm2(out1 + moe_output)
        
        return out2

class ProteinEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, 
                 num_experts, k, amino_acid_vocab_size, 
                 max_position, dropout_rate=0.1):
        super(ProteinEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(amino_acid_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_position, d_model)
        
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, d_ff, num_experts, k, dropout_rate) 
            for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]
        
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x  # (batch_size, input_seq_len, d_model)
