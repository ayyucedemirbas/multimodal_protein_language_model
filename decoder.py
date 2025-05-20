import tensorflow as tf
from layers import MultiheadAttention, MixtureOfExperts, positional_encoding

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, num_experts, k=2, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiheadAttention(d_model, num_heads, dropout_rate)
        self.mha2 = MultiheadAttention(d_model, num_heads, dropout_rate)
        self.moe = MixtureOfExperts(d_model, d_ff, num_experts, k, dropout_rate)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, enc_output, training=None, look_ahead_mask=None, padding_mask=None):
        attn1, attn_weights_block1 = self.mha1([inputs, inputs, inputs], look_ahead_mask, training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)
        
        attn2, attn_weights_block2 = self.mha2([out1, enc_output, enc_output], padding_mask, training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        moe_output = self.moe(out2, training=training)
        moe_output = self.dropout3(moe_output, training=training)
        out3 = self.layernorm3(out2 + moe_output)
        
        return out3, attn_weights_block1, attn_weights_block2

# Decoder for structure/function prediction
class ProteinDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, 
                 num_experts, k, target_vocab_size, 
                 max_position, dropout_rate=0.1):
        super(ProteinDecoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_position, d_model)
        
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, d_ff, num_experts, k, dropout_rate) 
            for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training=None, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)
            
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
