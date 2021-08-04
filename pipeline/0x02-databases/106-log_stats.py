#!/usr/bin/env python3
"""
Improves script 34-log_stats.py by adding the top 10 of the most present IPs
in the collection nginx of the database logs
"""


from pymongo import MongoClient


if __name__ == "__main__":
    """
    Provides stats about Nginx logs stored in MongoDB
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
    path_count = logs_coll.count_documents(
        {"method": "GET", "path": "/status"})
    print("{} status check".format(path_count))
    print("IPs:")
    ips = logs_coll.aggregate([
        {"$group": {"_id": "$ip", "count": {"$sum": 1}}}])
    ips_list = []
    for ip in ips:
        ips_list.append(ip)
    ips_list = sorted(ips_list, key=lambda i: i["count"], reverse=True)
    index = 0
    if len(ips_list) < 10:
        limit = len(ips_list)
    else:
        limit = 10
    while index < limit:
        ip = ips_list[index]["_id"]
        count = ips_list[index]["count"]
        print("\t{}: {}".format(ip, count))
        index += 1
