import tensorflow as tf

def preprocess_protein_sequence(sequence, max_length, vocab):
    #Convert protein sequence to integer tokens and pad.
    # Convert amino acid sequence to integers based on vocabulary
    tokens = [vocab.get(aa, 0) for aa in sequence]
    
    # Pad or truncate to max_length
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + [0] * (max_length - len(tokens))
    
    return tf.convert_to_tensor(tokens, dtype=tf.int32)

def preprocess_structure_data(structure_data, max_length, vocab):
    #Convert structure data to integer tokens and pad.
    # Convert structure elements to integers based on vocabulary
    tokens = [vocab.get(element, 0) for element in structure_data]
    
    # Add start and end tokens
    tokens = [1] + tokens + [2]  # 1=START, 2=END
    
    # Pad or truncate to max_length + 2 (for START/END tokens)
    if len(tokens) > max_length + 2:
        tokens = tokens[:max_length + 2]
    else:
        tokens = tokens + [0] * (max_length + 2 - len(tokens))
    
    return tf.convert_to_tensor(tokens, dtype=tf.int32)
