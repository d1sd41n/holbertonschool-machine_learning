#!/usr/bin/env python3
"""[summary]
"""


def list_all(mongo_collection):
    """[summary]

    Args:
        mongo_collection ([type]): [description]

    Returns:
        [type]: [description]
    """
    documents = []

    collection = mongo_collection.find()

    for doc in collection:
        documents.append(doc)

    return documents
