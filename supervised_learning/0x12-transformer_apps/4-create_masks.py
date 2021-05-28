#!/usr/bin/env python3
"""
Defines function that creates all masks for training/validation
"""


import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation

    parameters:
        inputs [tf.Tensor of shape (batch_size, seq_len_in)]:
            contains the input sentence
        target [tf.Tensor of shape (batch_size, seq_len_out)]:
            contains the target sentence
    returns:
        encoder_mask, combined_mask, decoder_mask
        encoder_mask [tf.Tensor of shape
                        (batch_size, 1, 1, seq_len_in)]:
            padding mask to be applied to the encoder
        combined_mask [tf.Tensor of shape
                        (batch_size, 1, seq_len_out, seq_len_out)]:
            mask used in the 1st attention block in the decoder to pad
                and mask future tokens in the input received by the decoder
            Maximum between look ahead mask and decoder target padding mask
        decoder_mask [tf.Tensor of shape (batch_size, 1, 1, seq_len_in)]:
            padding mask used in the 2nd attention block in the decoder
    """
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    batch_size, seq_len_out = target.shape

    look_ahead_mask = tf.linalg.band_part(tf.ones(
        (seq_len_out, seq_len_out)), -1, 0)
    look_ahead_mask = 1 - look_ahead_mask

    padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(look_ahead_mask, padding_mask)

    return encoder_mask, combined_mask, decoder_mask
