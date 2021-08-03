#!/usr/bin/env python3
"""
Provides some stats about Nginx logs stored in MongoDB
"""


from pymongo import MongoClient


if __name__ is "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx
    doc_count = logs.find().count()
    print(doc_count, " logs")
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        method_count = logs.find({"method": method}).count()
        print("\tmethod ", method, ": ", method_count)
    path_count = logs.find({"method": "GET", "path": "/status"}).count()
    print(path_count, " status check")
