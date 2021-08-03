#!/usr/bin/env python3
"""
Defines function that lists all documents in MongoDB collection
"""


def list_all(mongo_collection):
    """
    Lists all documents in given MongoDB collection

    parameters:
        mongo_collection: the collection to use

    returns:
        list of all documents or 0 if no documents found
    """
    all_docs = []
    collection = mongo_collection.find()
    for document in collection:
        all_docs.append(document)
    return all_docs
