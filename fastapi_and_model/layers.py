import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim

    def call(self, inputs):
        position_indices = tf.range(self.maxlen, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, self.embed_dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.embed_dim))
        
        # Create positional encoding matrix using TensorFlow operations
        sinusoids = tf.expand_dims(position_indices * div_term, -1)
        pos_enc = tf.concat([tf.sin(sinusoids), tf.cos(sinusoids)], axis=-1)
        pos_enc = tf.reshape(pos_enc, [1, self.maxlen, self.embed_dim])
        
        return inputs + pos_enc

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim
        })
        return config


@tf.keras.utils.register_keras_serializable()
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        })
        return config
