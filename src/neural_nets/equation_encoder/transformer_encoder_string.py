import numpy as np
import tensorflow as tf


class TransformerEncoderString(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.d_model = kwargs['embedding_dim_encoder_equation']
        self.num_layers = kwargs['num_layers']

        self.pos_embedding = PositionalEmbedding(
            vocab_size=kwargs['vocab_size'],
            d_model=kwargs['embedding_dim_encoder_equation'])

        self.enc_layers = [
            EncoderLayer(d_model=kwargs['embedding_dim_encoder_equation'],
                         num_heads=kwargs['num_heads'],
                         dff=kwargs['dff'],
                         dropout_rate=kwargs['dropout_rate'])
            for _ in range(kwargs['num_layers'])]
        self.dropout = tf.keras.layers.Dropout(kwargs['dropout_rate'])
        self.flatten = tf.keras.layers.Flatten()
        self.last_dense_layer = tf.keras.layers.Dense(16, dtype=np.float32)
        pass



    def call(self, x, training):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x, training=training)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        x  # Shape `(batch_size, seq_len, d_model)`.
        x = self.flatten(x, training=training)
        x = self.last_dense_layer(x)


        return x  # Shape `(batch_size, seq_len, d_model)`.


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True,  dtype=tf.float32)
        self.pos_encoding = self.positional_encoding(length=2024, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x, training):
        length = tf.shape(x)[1]
        x = self.embedding(x, training=training)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

    def positional_encoding(self, length, depth):
        """
  
        :param length: length od input
        :param depth: how many chanel to encode position
        :return:
        """
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

        angle_rates = 1 / (10000 ** depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention( dtype=tf.float32, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization( dtype=tf.float32)
        self.add = tf.keras.layers.Add( dtype=tf.float32)


class GlobalSelfAttention(BaseAttention):
    def call(self, x, training):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            training=training)
        x = self.add([x, attn_output])
        x = self.layernorm(
            x,
            training=training)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu', dtype=np.float32),
            tf.keras.layers.Dense(d_model, activation='relu', dtype=np.float32),
        ])
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add(dtype=np.float32)
        self.layer_norm = tf.keras.layers.LayerNormalization(dtype=np.float32)

    def call(self, x, training):
        x_seq = self.seq(x)
        x_seq = self.dropout(x_seq, training=training)
        x = self.add([x, x_seq])
        x = self.layer_norm(x, training=training)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff, dropout_rate=dropout_rate)

    def call(self, x, training):
        x = self.self_attention(x, training=training)
        x = self.ffn(x, training=training)
        return x
