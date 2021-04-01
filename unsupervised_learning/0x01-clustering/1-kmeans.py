#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """[summary]

    Args:
        X ([type]): [description]
        pi ([type]): [description]
        m ([type]): [description]
        S ([type]): [description]

    Returns:
        [type]: [description]
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if X.shape[1] != S.shape[1] or S.shape[1] != S.shape[2]:
        return (None, None)
    if X.shape[1] != m.shape[1] or m.shape[0] != S.shape[0]:
        return (None, None)
    if pi.shape[0] != m.shape[0]:
        return (None, None)
    if not np.isclose(np.sum(pi), 1):
        return None, None
    n, d = X.shape
    k = S.shape[0]
    t = np.zeros((k, n))
    for i in range(k):
        P = pdf(X, m[i], S[i])
        prior = pi[i]
        t[i] = prior * P
    g = t / np.sum(t, axis=0)
    likelihood = np.sum(np.log(np.sum(t, axis=0)))
    return g, likelihood
