#!/usr/bin/env python3
"""
Defines function that creates and trains a gensim word2vec model
"""


from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a gensim word2vec model

    parameters:
        sentences [list]:
            list of sentences to be trained on
        size [int]:
            dimensionality of the embedding layer
        min_count [int]:
            minimum number of occurances of a word for use in training
        window [int]:
            maximum distance between the current and predicted word
                within a sentence
        negative [int]:
            size of negative sampling
        cbow [boolean]:
            determines the training type
            True: CBOW
            False: Skip-gram
        iterations [int]:
            number of iterations to train over
        seed [int]:
            seed for the random number generator
        workers [int]:
            number of worker threads to train the model

    returns:
        the trained model
    """
    if cbow is True:
        cbow_flag = 0
    else:
        cbow_flag = 1
    model = Word2Vec(sentences=sentences,
                     size=size,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     sg=cbow_flag,
                     iter=iterations,
                     seed=seed,
                     workers=workers)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
