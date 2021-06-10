#!/usr/bin/env python3
"""
Defines function that finds a snippet of text within a reference document
to answer a question
"""


import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question

    parameters:
        question [string]:
            contains the question to answer
        reference [string]:
            contains the reference document from which to find the answer

    returns:
        [string]:
            contains the answer
        or None if no answer is found
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    quest_tokens = tokenizer.tokenize(question)
    refer_tokens = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + quest_tokens + ['[SEP]'] + refer_tokens + ['[SEP]']

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (
        1 + len(quest_tokens) + 1) + [1] * (len(refer_tokens) + 1)

    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids))

    outputs = model([input_word_ids, input_mask, input_type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if answer is None or answer is "" or question in answer:
        return None

    return answer
