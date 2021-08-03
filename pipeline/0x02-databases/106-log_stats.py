#!/usr/bin/env python3
"""[summary]
"""
from pymongo import MongoClient


if __name__ == "__main__":
    """[summary]
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx_collection = client.logs.nginx
    count_logs = nginx_collection.count_documents({})
    status = nginx_collection.count_documents({'method': 'GET',
                                               'path': '/status'})
    method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print('{} logs'.format(count_logs))
    print('Methods:')
    for m in method:
        n_data = nginx_collection.count_documents({'method': m})
        print('\tmethod {}: {}'.format(m, n_data))
    print('{} status check'.format(status))
    pipeline = [{'$unwind': '$ip'}, {"$sortByCount": '$ip'},
                {'$limit': 10}]
    a_ip = nginx_collection.aggregate(pipeline)
    data_ip = [i for i in a_ip]
    print('IPs:')
    for ip in data_ip:
        print('\t{}: {}'.format(ip['_id'], ip['count']))
