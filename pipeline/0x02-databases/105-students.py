#!/usr/bin/env python3
"""[summary]
"""


def top_students(mongo_collection):
    """[summary]

    Args:
        mongo_collection ([type]): [description]

    Returns:
        [type]: [description]
    """
    students = mongo_collection.find()
    students_list = []
    for student in students:
        topics = student["topics"]
        score = 0
        for topic in topics:
            score += topic["score"]
        score /= len(topics)
        student["averageScore"] = score
        students_list.append(student)
    return sorted(
        students_list, key=lambda i: i["averageScore"], reverse=True)
