#!/usr/bin/env python3
"""
Provides some stats about Nginx logs stored in MongoDB
"""


from pymongo import MongoClient


if __name__ == "__main__":
    """
    Gets stats about Nginx logs stored in MongoDB
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_coll = client.logs.nginx
    doc_count = logs_coll.count_documents({})
    print("{} logs".format(doc_count))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        method_count = logs_coll.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, method_count))
    filter_path = {"method": "GET", "path": "/status"}
    path_count = logs_coll.count_documents(filter_path)
    print("{} status check".format(path_count))
