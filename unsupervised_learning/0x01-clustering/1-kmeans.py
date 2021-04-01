#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def distances(a, b):
    """[summary]

    Args:
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [description]
    """
    b = b[np.newaxis, :]
    a = a[:, np.newaxis, :]
    diff = a - b
    dist = np.linalg.norm(diff, axis=-1, keepdims=False)
    return dist


def initialize(X, k):
    """[summary]

    Args:
        X ([type]): [description]
        k ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    centroids = np.random.uniform(mins, maxs, size=(k, X.shape[1]))
    return centroids


def kmeans(X, k, iterations=1000):
    """[summary]

    Args:
        X ([type]): [description]
        k ([type]): [description]
        iterations (int, optional): [description]. Defaults to 1000.

    Returns:
        [type]: [description]
    """
    # (k, d)
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None
    C = initialize(X, k)
    if C is None:
        return None, None
    k, d = C.shape
    n, _ = X.shape
    dist = np.zeros(n)
    for i in range(iterations):
        dists = distances(X, C)
        clss = np.argmin(dists, axis=1).reshape(-1)
        C_changed = False
        for j in range(k):
            clust = X[clss == j]
            if len(clust) == 0:
                mean_j = initialize(X, 1)[0]
            else:
                mean_j = clust.mean(axis=0)

            if (mean_j != C[j]).all():
                C[j] = mean_j
                C_changed = True
        if not C_changed:
            break
    return C, clss
