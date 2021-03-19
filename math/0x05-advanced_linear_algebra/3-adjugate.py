#!/usr/bin/env python3
"""[summary]
"""


def minor(matrix):
    """[summary]

    Args:
        matrix ([type]): [description]

    Raises:
        TypeError: [description]
        TypeError: [description]
        TypeError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if not matrix:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix[0]) == 1:
        return [[1]]

    minor_m = []
    for r in range(len(matrix)):
        min_r = []
        for c in range(len(matrix)):
            row = [matrix[i] for i in range(len(matrix)) if i != r]
            sub_m = []

            for r_aux in row:
                aux = []
                for col in range(len(matrix)):
                    if col != c:
                        aux.append(r_aux[col])
                sub_m.append(aux)

            det = determinant(sub_m)
            min_r.append(det)
        minor_m.append(min_r)
    return minor_m


def determinant(matrix):
    """[summary]

    Args:
        matrix ([type]): [description]

    Raises:
        TypeError: [description]
        TypeError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i, j in enumerate(matrix[0]):
        _mrows = [r for r in matrix[1:]]
        aux_sub = []
        for row in _mrows:
            aux = []
            for c in range(len(matrix)):
                if c != i:
                    aux.append(row[c])
            aux_sub.append(aux)
        det += j * (-1) ** i * determinant(aux_sub)
    return det


def cofactor(matrix):
    """[summary]

    Args:
        matrix ([type]): [description]

    Returns:
        [type]: [description]
    """
    cofactor = minor(matrix)
    for row in range(len(cofactor)):
        for col in range(len(cofactor[0])):
            cofactor[row][col] *= ((-1) ** (row + col))
    return cofactor


def adjugate(matrix):
    """[summary]

    Args:
        matrix ([type]): [description]

    Returns:
        [type]: [description]
    """
    cofacto_r = cofactor(matrix)
    adjugate = []
    for row in range(len(matrix)):
        adjugate.append([])
        for col in range(len(matrix)):
            adjugate[row].append(cofacto_r[col][row])
    return adjugate
