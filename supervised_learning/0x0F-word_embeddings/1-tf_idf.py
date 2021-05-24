#!/usr/bin/env python3
"""
Defines function that creates a TF-IDF embedding
"""


from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding

    parameters:
        sentences [list]:
            list of sentences to analyze

        vocab [list]:
            list of vocabulary words to use for analysis
            if None, all words within sentences should be used

    returns:
        embeddings,features:
            embeddings [numpy.ndarray of shape (s, f)]:
                contains the embeddings
                s: number of sentences in sentences
                f: number of features analyzed
            features [list]:
                list of features used for embeddings
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embeddings = x.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
