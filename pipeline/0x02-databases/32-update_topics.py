#!/usr/bin/env python3
"""[summary]
"""


def update_topics(mongo_collection, name, topics):
    """[summary]

    Args:
        mongo_collection ([type]): [description]
        name ([type]): [description]
        topics ([type]): [description]
    """
    newvalues = {"$set": {"topics": topics}}
    mongo_collection.update_many({"name": name}, newvalues)
