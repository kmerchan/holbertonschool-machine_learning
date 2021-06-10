#!/usr/bin/env python3
"""
Defines function that performs semantic search on a corpus of documents
"""


import numpy as np
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents

    parameters:
        corpus_path [string]:
            the path to the corpus of reference documents on which
                to perform semantic search
        sentence [string]:
            the sentence from which to perform semantic search

    returns:
        [string]:
            the reference text of the document most similar to given sentence
    """
    documents = [sentence]

    for filename in os.listdir(corpus_path):
        if filename.endswith(".md") is False:
            continue
        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
            documents.append(f.read())

    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")

    embeddings = model(documents)

    correlation = np.inner(embeddings, embeddings)

    closest = np.argmax(correlation[0, 1:])

    similar = documents[closest + 1]

    return similar
