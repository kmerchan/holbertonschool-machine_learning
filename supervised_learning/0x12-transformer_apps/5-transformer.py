#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to create transformer network
"""


import numpy as np
import tensorflow as tf


def get_angle(pos, i, dm):
    """
    Calculates the angles for the following formulas for positional encoding:

    PE(pos, 2i) = sin(pos / 10000^(2i / dm))
    PE(pos, 2i + 1) = cos(pos / 10000^(2i / dm))
    """
    angle_rates = 1 / (10000 ** (i / dm))
    return pos * angle_rates


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    parameters:
        max_seq_len [int]:
            represents the maximum sequence length
        dm: model depth

    returns:
        [numpy.ndarray of shape (max_seq_len, dm)]:
            contains the positional encoding vectors
    """
    positional_encoding = np.zeros([max_seq_len, dm])

    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            # sin for even indices of positional_encoding
            positional_encoding[pos, i] = np.sin(get_angle(pos, i, dm))
            # cos for odd indices of positional_encoding
            positional_encoding[pos, i + 1] = np.cos(get_angle(pos, i, dm))
    return positional_encoding


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention

    parameters:
        Q [tensor with last two dimensions as (..., seq_len_q, dk)]:
            contains the query matrix
        K [tensor with last two dimensions as (..., seq_len_v, dk)]:
            contains the key matrix
        V [tensor with last two dimensions as (..., seq_len_v, dv)]:
            contains the value matrix
        mask [tensor that can be broadcast into (..., seq_len_q, seq_len_v)]:
            contains the optional mask, or defaulted to None

    returns:
        outputs, weights:
            outputs [tensor with last two dimensions as (..., seq_len_q, dv)]:
                contains the scaled dot product attention
            weights [tensor with last two dimensions as
                    (..., seq_len_q, seq_len_v)]:
                contains the attention weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    # scale matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add mask to scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # normalize softmax on last axis so all scores add up to 1
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # calculate outputs
    outputs = tf.matmul(weights, V)
    return outputs, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class to perform multi-head attention

    class constructor:
        def __init__(self, dm, h)

    public instance attribute:
        h: number of heads
        dm: the dimensionality of the model
        depth: the depth of each attention head
        Wq: a Dense layer with dm units, used to generate the query matrix
        Wk: a Dense layer with dm units, used to generate the key matrix
        Wv: a Dense layer with dm units, used to generate the value matrix
        linear: a Dense layer with dm units, used to generate attention output

    public instance methods:
        def call(self, Q, K, V, mask):
            generates the query, key, and value matrices and
                outputs the scaled dot product attention
    """
    def __init__(self, dm, h):
        """
        Class constructor

        parameters:
            dm [int]:
                represents the dimensionality of the model
            h [int]:
                represents the number of heads

        sets the public instance attributes:
            h: number of heads
            dm: the dimensionality of the model
            depth: the depth of each attention head
            Wq: a Dense layer with dm units, used to generate the query matrix
            Wk: a Dense layer with dm units, used to generate the key matrix
            Wv: a Dense layer with dm units, used to generate the value matrix
            linear: a Dense layer with dm units,
                used to generate attention output
        """
        if type(dm) is not int:
            raise TypeError(
                "dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError(
                "h must be int representing number of heads")
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch):
        """
        Splits the last dimension of tensor into (h, dm) and
            transposes the result so the shape is (batch, h, seq_len, dm)
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def call(self, Q, K, V, mask):
        """
        Generates the query, key, and value matrices and
            outputs the scaled dot product attention

        parameters:
            Q [tensor of shape (batch, seq_len_q, dk)]:
                contains the input to generate the query matrix
            K [tensor of shape (batch, seq_len_v, dk)]:
                contains the input to generate the key matrix
            V [tensor of shape (batch, seq_len_v, dv)]:
                contains the input to generate the value matrix
            mask [always None]

        returns:
            outputs, weights:
                outputs [tensor with last two dimensions (..., seq_len_q, dm)]:
                    contains the scaled dot product attention
                weights [tensor with last dimensions
                        (..., h, seq_len_q, seq_len_v)]:
                    contains the attention weights
        """
        # batch = Q.get_shape().as_list()[0]
        batch = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch)
        k = self.split_heads(k, batch)
        v = self.split_heads(v, batch)

        attention, weights = sdp_attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch, -1, self.dm))
        outputs = self.linear(concat_attention)

        return outputs, weights


class EncoderBlock(tf.keras.layers.Layer):
    """
    Class to create an encoder block for a transformer

    class constructor:
        def __init__(self, dm, h, hidden, drop_rate=0.1)

    public instance attribute:
        mha: MultiHeadAttention layer
        dense_hidden: the hidden dense layer with hidden units, relu activation
        dense_output: the output dense layer with dm units
        layernorm1: the first layer norm layer, with epsilon=1e-6
        layernorm2: the second layer norm layer, with epsilon=1e-6
        drouput1: the first dropout layer
        dropout2: the second dropout layer

    public instance method:
        call(self, x, training, mask=None):
            calls the encoder block and returns the block's output
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor

        parameters:
            dm [int]:
                represents the dimensionality of the model
            h [int]:
                represents the number of heads
            hidden [int]:
                represents the number of hidden units in fully connected layer
            drop_rate [float]:
                the dropout rate

        sets the public instance attributes:
            mha: MultiHeadAttention layer
            dense_hidden: the hidden dense layer with hidden units, relu activ.
            dense_output: the output dense layer with dm units
            layernorm1: the first layer norm layer, with epsilon=1e-6
            layernorm2: the second layer norm layer, with epsilon=1e-6
            drouput1: the first dropout layer
            dropout2: the second dropout layer
        """
        if type(dm) is not int:
            raise TypeError(
                "dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError(
                "h must be int representing number of heads")
        if type(hidden) is not int:
            raise TypeError(
                "hidden must be int representing number of hidden units")
        if type(drop_rate) is not float:
            raise TypeError(
                "drop_rate must be float representing dropout rate")
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Calls the encoder block and returns the block's output

        parameters:
            x [tensor of shape (batch, input_seq_len, dm)]:
                contains the input to the encoder block
            training [boolean]:
                determines if the model is in training
            mask:
                mask to be applied for multi-head attention

        returns:
            [tensor of shape (batch, input_seq_len, dm)]:
                contains the block's output
        """
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        output1 = self.layernorm1(x + attention_output)

        dense_output = self.dense_hidden(output1)
        ffn_output = self.dense_output(dense_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output1 + ffn_output)

        return output2


class DecoderBlock(tf.keras.layers.Layer):
    """
    Class to create a decoder block for a transformer

    class constructor:
        def __init__(self, dm, h, hidden, drop_rate=0.1)

    public instance attribute:
        mha1: the first MultiHeadAttention layer
        mha2: the second MultiHeadAttention layer
        dense_hidden: the hidden dense layer with hidden units, relu activation
        dense_output: the output dense layer with dm units
        layernorm1: the first layer norm layer, with epsilon=1e-6
        layernorm2: the second layer norm layer, with epsilon=1e-6
        layernorm3: the third layer norm layer, with epsilon=1e-6
        drouput1: the first dropout layer
        dropout2: the second dropout layer
        dropout3: the third dropout layer

    public instance method:
        def call(self, x, encoder_output, training, look_ahead_mask,
                    padding_mask):
            calls the decoder block and returns the block's output
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor

        parameters:
            dm [int]:
                represents the dimensionality of the model
            h [int]:
                represents the number of heads
            hidden [int]:
                represents the number of hidden units in fully connected layer
            drop_rate [float]:
                the dropout rate

        sets the public instance attributes:
            mha1: the first MultiHeadAttention layer
            mha2: the second MultiHeadAttention layer
            dense_hidden: the hidden dense layer with hidden units, relu activ.
            dense_output: the output dense layer with dm units
            layernorm1: the first layer norm layer, with epsilon=1e-6
            layernorm2: the second layer norm layer, with epsilon=1e-6
            layernorm3: the third layer norm layer, with epsilon=1e-6
            drouput1: the first dropout layer
            dropout2: the second dropout layer
            dropout3: the third dropout layer
        """
        if type(dm) is not int:
            raise TypeError(
                "dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError(
                "h must be int representing number of heads")
        if type(hidden) is not int:
            raise TypeError(
                "hidden must be int representing number of hidden units")
        if type(drop_rate) is not float:
            raise TypeError(
                "drop_rate must be float representing dropout rate")
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Calls the decoder block and returns the block's output

        parameters:
            x [tensor of shape (batch, target_seq_len, dm)]:
                contains the input to the decoder block
            encoder_output [tensor of shape (batch, input_seq_len, dm)]:
                contains the output of the encoder
            training [boolean]:
                determines if the model is in training
            look_ahead_mask:
                mask to be applied to the first multi-head attention
            padding_mask:
                mask to be applied to the second multi-head attention

        returns:
            [tensor of shape (batch, target_seq_len, dm)]:
                contains the block's output
        """
        attention_output1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention_output1 = self.dropout1(attention_output1, training=training)
        output1 = self.layernorm1(x + attention_output1)

        attention_output2, _ = self.mha2(output1, encoder_output,
                                         encoder_output, padding_mask)
        attention_output2 = self.dropout2(attention_output2, training=training)
        output2 = self.layernorm2(output1 + attention_output2)

        dense_output = self.dense_hidden(output2)
        ffn_output = self.dense_output(dense_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        output3 = self.layernorm3(output2 + ffn_output)

        return output3


class Encoder(tf.keras.layers.Layer):
    """
    Class to create the encoder for a transformer

    class constructor:
        def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                        drop_rate=0.1)

    public instance attribute:
        N: the number of blocks in the encoder
        dm: the dimensionality of the model
        embedding: the embedding layer for the inputs
        positional_encoding [numpy.ndarray of shape (max_seq_len, dm)]:
            contains the positional encodings
        blocks [list of length N]:
            contains all the EncoderBlocks
        dropout: the dropout layer, to be applied to the positional encodings

    public instance method:
        call(self, x, training, mask):
            calls the encoder and returns the encoder's output
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor

        parameters:
            N [int]:
                represents the number of blocks in the encoder
            dm [int]:
                represents the dimensionality of the model
            h [int]:
                represents the number of heads
            hidden [int]:
                represents the number of hidden units in fully connected layer
            input_vocab [int]:
                represents the size of the input vocabulary
            max_seq_len [int]:
                represents the maximum sequence length possible
            drop_rate [float]:
                the dropout rate

        sets the public instance attributes:
            N: the number of blocks in the encoder
            dm: the dimensionality of the model
            embedding: the embedding layer for the inputs
            positional_encoding [numpy.ndarray of shape (max_seq_len, dm)]:
                contains the positional encodings
            blocks [list of length N]:
                contains all the EncoderBlocks
            dropout: the dropout layer, applied to the positional encodings
        """
        if type(N) is not int:
            raise TypeError(
                "N must be int representing number of blocks in the encoder")
        if type(dm) is not int:
            raise TypeError(
                "dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError(
                "h must be int representing number of heads")
        if type(hidden) is not int:
            raise TypeError(
                "hidden must be int representing number of hidden units")
        if type(input_vocab) is not int:
            raise TypeError(
                "input_vocab must be int representing size of input vocab")
        if type(max_seq_len) is not int:
            raise TypeError(
                "max_seq_len must be int representing max sequence length")
        if type(drop_rate) is not float:
            raise TypeError(
                "drop_rate must be float representing dropout rate")
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for block in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Calls the encoder and returns the encoder's output

        parameters:
            x [tensor of shape (batch, input_seq_len, dm)]:
                contains the input to the encoder
            training [boolean]:
                determines if the model is in training
            mask:
                mask to be applied for multi-head attention

        returns:
            [tensor of shape (batch, input_seq_len, dm)]:
                contains the encoder output
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """
    Class to create the decoder for a transformer

    class constructor:
        def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                        drop_rate=0.1)

    public instance attribute:
        N: the number of blocks in the encoder
        dm: the dimensionality of the model
        embedding: the embedding layer for the targets
        positional_encoding [numpy.ndarray of shape (max_seq_len, dm)]:
            contains the positional encodings
        blocks [list of length N]:
            contains all the DecoderBlocks
        dropout: the dropout layer, to be applied to the positional encodings

    public instance method:
        def call(self, x, encoder_output, training, look_ahead_mask,
                    padding_mask):
            calls the decoder and returns the decoder's output
    """
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor

        parameters:
            N [int]:
                represents the number of blocks in the encoder
            dm [int]:
                represents the dimensionality of the model
            h [int]:
                represents the number of heads
            hidden [int]:
                represents the number of hidden units in fully connected layer
            target_vocab [int]:
                represents the size of the target vocabulary
            max_seq_len [int]:
                represents the maximum sequence length possible
            drop_rate [float]:
                the dropout rate

        sets the public instance attributes:
            N: the number of blocks in the encoder
            dm: the dimensionality of the model
            embedding: the embedding layer for the targets
            positional_encoding [numpy.ndarray of shape (max_seq_len, dm)]:
                contains the positional encodings
            blocks [list of length N]:
                contains all the DecoderBlocks
            dropout: the dropout layer,
                to be applied to the positional encodings
        """
        if type(N) is not int:
            raise TypeError(
                "N must be int representing number of blocks in the encoder")
        if type(dm) is not int:
            raise TypeError(
                "dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError(
                "h must be int representing number of heads")
        if type(hidden) is not int:
            raise TypeError(
                "hidden must be int representing number of hidden units")
        if type(target_vocab) is not int:
            raise TypeError(
                "target_vocab must be int representing size of target vocab")
        if type(max_seq_len) is not int:
            raise TypeError(
                "max_seq_len must be int representing max sequence length")
        if type(drop_rate) is not float:
            raise TypeError(
                "drop_rate must be float representing dropout rate")
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for block in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Calls the decoder and returns the decoder's output

        parameters:
            x [tensor of shape (batch, target_seq_len, dm)]:
                contains the input to the decoder
            encoder_output [tensor of shape (batch, input_seq_len, dm)]:
                contains the output of the encoder
            training [boolean]:
                determines if the model is in training
            look_ahead_mask:
                mask to be applied to first multi-head attention
            padding_mask:
                mask to be applied to second multi-head attention

        returns:
            [tensor of shape (batch, target_seq_len, dm)]:
                contains the decoder output
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)
        return x


class Transformer(tf.keras.layers.Layer):
    """
    Class to create the transformer network

    class constructor:
        def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                     max_seq_input, max_seq_target, drop_rate=0.1)

    public instance attributes:
        encoder: the encoder layer
        decoder: the decoder layer
        linear: the Dense layer with target_vocab units

    public instance method:
        def call(self, inputs, target, training, encoder_mask,
                    look_ahead_mask, decoder_mask):
            calls the transformer network and returns the transformer output
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor

        parameters:
            N [int]:
                represents the number of blocks in the encoder and decoder
            dm [int]:
                represents the dimensionality of the model
            h [int]:
                represents the number of heads
            hidden [int]:
                represents the number of hidden units in fully connected layer
            input_vocab [int]:
                represents the size of the input vocabulary
            target_vocab [int]:
                represents the size of the target vocabulary
            max_seq_input [int]:
                represents the maximum sequence length possible for input
            max_seq_target [int]:
                represents the maximum sequence length possible for target
            drop_rate [float]:
                the dropout rate

        sets the public instance attributes:
            encoder: the encoder layer
            decoder: the decoder layer
            linear: the Dense layer with target_vocab units
        """
        if type(N) is not int:
            raise TypeError(
                "N must be int representing number of blocks in the encoder")
        if type(dm) is not int:
            raise TypeError(
                "dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError(
                "h must be int representing number of heads")
        if type(hidden) is not int:
            raise TypeError(
                "hidden must be int representing number of hidden units")
        if type(input_vocab) is not int:
            raise TypeError(
                "input_vocab must be int representing size of input vocab")
        if type(target_vocab) is not int:
            raise TypeError(
                "target_vocab must be int representing size of target vocab")
        if type(max_seq_input) is not int:
            raise TypeError(
                "max_seq_input must be int representing max length for input")
        if type(max_seq_target) is not int:
            raise TypeError(
                "max_seq_target must be int representing max len for target")
        if type(drop_rate) is not float:
            raise TypeError(
                "drop_rate must be float representing dropout rate")
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Calls the transformer network and returns the transformer output

        parameters:
            inputs [tensor of shape (batch, input_seq_len)]:
                contains the inputs
            target [tensor of shape (batch, target_seq_len)]:
                contains the target
            training [boolean]:
                determines if the model is in training
            encoder_mask:
                padding mask to be applied to the encoder
            look_ahead_mask:
                look ahead mask to be applied to the decoder
            decoder_mask:
                padding mask to be applied to the decoder

        returns:
            [tensor of shape (batch, target_seq_len, target_vocab)]:
                contains the transformer output
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        final_output = self.linear(decoder_output)
        return final_output
