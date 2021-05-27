#!/usr/bin/env python3
"""
Defines class Dataset that loads and preps a dataset for machine translation
"""


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Loads and preps a dataset for machine translation

    class constructor:
        def __init__(self, batch_size, max_len)

    public instance attributes:
        data_train:
            contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset train split, loaded as_supervided
        data_valid:
            contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset validate split, loaded as_supervided
        tokenizer_pt:
            the Portuguese tokenizer created from the training set
        tokenizer_en:
            the English tokenizer created from the training set

    instance method:
        def tokenize_dataset(self, data):
            that creates sub-word tokenizers for our dataset
        def encode(self, pt, en):
            that encodes a translation into tokens
        def tf_encode(self, pt, en):
            that acts as a TensorFlow wrapper for the encode method
    """
    def __init__(self, batch_size, max_len):
        """
        Class constructor

        parameters:
            batch_size [int]:
                the batch size for training/validation
            max_len [int]:
                the maximum number of tokens allowed per example sentence

        Sets the public instance attributes:
            data_train:
                contains the ted_hrlr_translate/pt_to_en
                    tf.data.Dataset train split, loaded as_supervided
            data_valid:
                contains the ted_hrlr_translate/pt_to_en
                    tf.data.Dataset validate split, loaded as_supervided
            tokenizer_pt:
                the Portuguese tokenizer created from the training set
            tokenizer_en:
                the English tokenizer created from the training set
        """
        data_train = tfds.load("ted_hrlr_translate/pt_to_en",
                               split="train",
                               as_supervised=True)
        data_valid = tfds.load("ted_hrlr_translate/pt_to_en",
                               split="validation",
                               as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train)

        # set attributes to encoded data
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

        def filter_max_length(x, y, max_len=max_len):
            """
            Filters data by max_len
            """
            filtered = tf.logical_and(tf.size(x) <= max_len,
                                      tf.size(y) <= max_len)
            return filtered

        # filter training and validation by max_len number of tokens
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_valid = self.data_valid.filter(filter_max_length)

        # increase performance by caching training dataset
        self.data_train = self.data_train.cache()

        # shuffle the training dataset
        data_size = sum(1 for data in self.data_train)
        self.data_train = self.data_train.shuffle(data_size)

        # split training and validation datasets into padded batches
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_valid = self.data_valid.padded_batch(batch_size)

        # increase performance by prefetching training dataset
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

    def tokenize_dataset(self, data):
        """
        Creates sub_word tokenizers for our dataset

        parameters:
            data [tf.data.Dataset]:
                dataset to use whose examples are formatted as tuple (pt, en)
                pt [tf.Tensor]:
                    contains the Portuguese sentence
                en [tf.Tensor]:
                    contains the corresponding English sentence
        returns:
            tokenizer_pt, tokenizer_en:
                tokenizer_pt: the Portuguese tokenizer
                tokenizer_en: the English tokenizer
        """
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=(2 ** 15))
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=(2 ** 15))
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens

        parameters:
            pt [tf.Tensor]:
                contains the Portuguese sentence
            en [tf.Tensor]:
                contains the corresponding English sentence
        returns:
            pt_tokens, en_tokens:
                pt_tokens [np.ndarray]: the Portuguese tokens
                en_tokens [np.ndarray]: the English tokens
        """
        pt_start_index = self.tokenizer_pt.vocab_size
        pt_end_index = pt_start_index + 1
        en_start_index = self.tokenizer_en.vocab_size
        en_end_index = en_start_index + 1
        pt_tokens = [pt_start_index] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_end_index]
        en_tokens = [en_start_index] + self.tokenizer_en.encode(
            en.numpy()) + [en_end_index]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a TensorFlow wrapper for the encode method
            to return tensors instead of numpy arrays

        parameters:
            pt [tf.Tensor]:
                contains the Portuguese sentence
            en [tf.Tensor]:
                contains the corresponding English sentence

        returns:
            pt [tf.Tensor]: encoded Portuguese sentence
            en [tf.Tensor]: encoded English sentence
        """
        pt_encoded, en_encoded = tf.py_function(func=self.encode,
                                                inp=[pt, en],
                                                Tout=[tf.int64, tf.int64])
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])
        return pt_encoded, en_encoded
