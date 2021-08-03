#!/usr/bin/env python3
"""[summary]
"""


def insert_school(mongo_collection, **kwargs):
    """[summary]

    Args:
        mongo_collection ([type]): [description]

    Returns:
        [type]: [description]
    """
    id_ = mongo_collection.insert_one(kwargs).inserted_id
    return id_
