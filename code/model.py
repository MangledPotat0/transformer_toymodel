# -*- conding: utf-8 -*-

"""
Model definition script based on the example:
https://keras.io/examples/nlp/ner_transformers/
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import ops
from keras import layers

class TransformerBlock(layers.Layer):
    """
    Transformer block consisting of Multi-head attention layers
    and feed forward.

    Args:
        embed_dim (int): Dimension of the input embeddings.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimension of the feed-forward network.
        rate (float, optional): Dropout rate (default is 0.1).

    Attributes:
        att (keras.layers.MultiHeadAttention): Multi-head self-attention layer.
        ffn (keras.Sequential): Feed-forward neural network.
        layernorm1 (keras.layers.LayerNormalization): Layer normalization for the first sub-layer.
        layernorm2 (keras.layers.LayerNormalization): Layer normalization for the second sub-layer.
        dropout1 (keras.layers.Dropout): Dropout layer for the first sub-layer.
        dropout2 (keras.layers.Dropout): Dropout layer for the second sub-layer.

    Methods:
        call(inputs, training=False):
            Applies the Transformer block to the input tensor.

    Example:
        transformer_block = TransformerBlock(embed_dim=512, num_heads=8, ff_dim=2048)
        output = transformer_block(input_tensor
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_neads=num_neads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        """
        Call stack for the TransformerBlock layer. The sequence of model is
        multi-head attention -> dropout -> layer normalization with residual
        connection -> feed-forward -> dropout -> layer norm with residual.

        Applies the Transformer block to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool, optional): Whether the model is in training mode (default is False).

        Returns:
            tf.Tensor: Output tensor after applying the Transformer block.
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.drouput1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = self.layernorm2(out1 + ffn_output)
        return ffn_output

class TokenAndPoisitionEmbedding(layers.Layer):
    """
    A custom layer representing token and position embeddings for a Transformer model.

    Args:
        maxlen (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Dimension of the embedding vectors.

    Attributes:
        token_emb (keras.layers.Embedding): Token embedding layer.
        pos_emb (keras.layers.Embedding): Position embedding layer.

    Methods:
        call(inputs):
            Computes token and position embeddings for the input tensor.

    Example:
        embedding_layer = TokenAndPositionEmbedding(maxlen=100, vocab_size=5000, embed_dim=256)
        input_tensor = tf.keras.Input(shape=(100,))
        embeddings = embedding_layer(input_tensor)
    """

    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim
        )

    def call(self, inputs):
        """
        Computes token and position embeddings for the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor representing token indices.

        Returns:
            tf.Tensor: Combined token and position embeddings.
        """
        maxlen = ops.shape(inputs)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        embeddings = token_embeddings + position_embeddings
        return embeddings

class NERModel(keras.Model): # pylint: ignore
    def __init__(
            self, num_tags, vocab_size, maxlen=128, embed_dim=32,
            num_heads=2, ff_dim=32
    ):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(
            maxlen, vocab_size, embed_dim
        )
        self.transformer_block = TransformerBlock(
            embed_dim, num_heads, ff_dim
        )
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(ff_dim, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x

# EOF
