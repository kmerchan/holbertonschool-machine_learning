#!/usr/bin/env python3
"""
Defines function that changes all topics of a school document based on name
"""


def update_topics(mongo_collection, name, topics):
    """
    Changes all topics of a school document based on the name

    parameters:
        mongo_collection [pymongo]:
            the MongoDB collection to use
        name [string]:
            the school name to update
        topics [list of strings]:
            list of topics approached in the school
    """
    mongo_collection.update({school: name},
                            {'$set': {topics: topics}}, false, true)
