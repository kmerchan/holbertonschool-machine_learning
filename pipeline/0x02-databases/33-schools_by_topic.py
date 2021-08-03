#!/usr/bin/env python3
"""
Defines function that returns the list of schools with a specific topic
"""


def schools_by_topic(mongo_collection, topic):
    """
    Finds list of all schools with a specific topic

    parameters:
        mongo_collection [pymongo]:
            the MongoDB collection to use
        topic [string]:
            the topic to search for

    returns:
        list of schools with the given topic
    """
    schools = []
    documents = mongo_collection.find({'topics': {'$all': [topic]}})
    for doc in documents:
        schools.append(doc)
    return schools
