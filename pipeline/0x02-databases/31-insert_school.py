#!/usr/bin/env python3
"""
Defines function that inserts a new document in a MongoDB collection
   based on kwargs
"""


def insert_school(mongo_collection, **kwargs):
    """
    Inserts a new document in a MongoDB collection based on kwargs

    parameters:
        kwargs: the new document to add

    returns:
        the new _id
    """
    document = mongo_collection.insert_one(kwargs)
    return (document.inserted_id)
