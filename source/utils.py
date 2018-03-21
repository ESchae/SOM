import numpy as np


def euclidean_distance(vector_x, vector_y):
    """ Calculate euclidean distance between two vectors.

    >>> vector_x = np.array([0.0, 0.0, 0.0])
    >>> euclidean_distance(vector_x, np.array([0.0, 0.0, 0.0]))
    0.0
    >>> euclidean_distance(vector_x, np.array([2.0, 2.0, 1.0]))
    3.0
    >>> euclidean_distance(vector_x, np.array([1.0, 1.0, 1.0]))
    1.7320508075688772

    """
    distance = 0
    length = min(len(vector_x), len(vector_y))
    for i in range(length):
        distance += (vector_x[i] - vector_y[i]) ** 2
    return np.sqrt(distance)


def get_neighbor_indices(index, shape):
    """ Given an index in a matrix return all neighbor indices.

    >>> get_neighbor_indices(index=(0, 0), shape=(5, 2))
    [(0, 1), (1, 0), (1, 1)]
    >>> get_neighbor_indices(index=(2, 2), shape=(3, 3))
    [(1, 1), (1, 2), (2, 1)]
    >>> get_neighbor_indices(index=(1, 1), shape=(3, 3))
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]

    :param index [tuple]: index in a matrix
    :param shape [tuple]: shape of the matrix
    :return [list of tuples]: neighbor indices
    """
    neighbor_indices = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            neighbor_i = i + index[0]
            neighbor_j = j + index[1]
            if (0 <= neighbor_i < shape[0]) and (0 <= neighbor_j < shape[1]):
                neighbor_index = (neighbor_i, neighbor_j)
                if neighbor_index != index:  # do not include given index itself
                    neighbor_indices.append(neighbor_index)
    return neighbor_indices
