#!/usr/bin/env python3
"""[summary]
"""


def schools_by_topic(mongo_collection, topic):
    """[summary]

    Args:
        mongo_collection ([type]): [description]
        topic ([type]): [description]

    Returns:
        [type]: [description]
    """
    match = []

    results = mongo_collection.find({"topics": {"$all": [topic]}})

    for result in results:
        match.append(result)

    return match
