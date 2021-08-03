#!/usr/bin/env python3
"""
Provides some stats about Nginx logs stored in MongoDB
"""


from pymongo import MongoClient


if __name__ is "__main__":
    """
    Provides stats about Nginx logs stored in MongoDB
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_coll = client.logs.nginx
    doc_count = logs_coll.count_documents()
    print("{} logs".format(doc_count)
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        method_count = logs_coll.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, method_count))
    path_count = logs_coll.count_documents(
        {"method": "GET", "path": "/status"})
    print("{} status check".format(path_count))
