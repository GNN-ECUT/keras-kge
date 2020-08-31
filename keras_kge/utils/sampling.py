from random import choices

import numpy as np


def sample_corrupted_triples(positive_triples: np.ndarray, dim: int, nb_samples: int) -> np.ndarray:
    """
     Sample corrupted triples, given a set of input triples.

    :param positive_triples: numpy array of shape (n, 3) containing the n input triples
    :param dim: dimension to be corrupted < 0 | 1 | 2 >
    :param nb_samples: number of corrupted triples per input triple
    :return: numpy array of shape (n * nb_samples, 3) containing the corrupted triples
    """
    assert dim < 3

    max_sample = max(positive_triples[:, dim])

    negative_triples = []

    for triple in positive_triples:

        corrupted = choices([i for i in range(0, max_sample) if i is not triple[dim]], k=nb_samples)

        corrupted_triples = np.array([triple] * nb_samples)
        corrupted_triples[:, dim] = corrupted

        negative_triples.append(corrupted_triples)

    return np.vstack(negative_triples)
