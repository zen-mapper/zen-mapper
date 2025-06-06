"""
Methods for generating synthetic topological data
"""

import sys

import numpy as np

type Seed = (
    None | int | np.random.SeedSequence | np.random.BitGenerator | np.random.Generator
)


def unit_simplex(
    dim: int,
    num_samples: int = 1,
    closed: bool = True,
    seed: Seed = None,
) -> np.ndarray:
    """Sample uniformly from the unit n-simplex

    This uses the modified Kraemer algorithm as described in Smith & Tromble _[1].

    Parameters
    ----------
    dim : int
        Dimension of simplex to sample from.
    num_samples : int
        Number of points to sample.
    closed : bool
        If true, the region sampled includes the boundary of the simplex.
    seed : Seed
        The seed for the random number generator.

    Returns
    -------
    `np.array` of shape (num_samples, dim+1), with the rows corresponding to
    individual samples.

    .. [1] Noah A. Smith and Roy W. Tromble, "Sampling Uniformly from the Unit
    Simplex," 2004.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be > 0")

    if dim < 0:
        raise ValueError("dim must be > 0")

    M = sys.maxsize
    rng = np.random.default_rng(seed)

    n = dim + 1

    result = np.zeros((num_samples, n))

    for i in range(num_samples):
        result[i, 1:] = rng.choice(M - 1, replace=False, size=n - 1) + 1

    result.sort(axis=1)
    result = np.diff(result, append=M, axis=1)

    if not closed:
        result /= M
    else:
        result = (result - 1) / (M - n)

    return result


def simplex(
    simplex: np.ndarray,
    num_samples: int = 1,
    closed: bool = True,
    seed: Seed = None,
) -> np.ndarray:
    """Sample uniformly from an arbitrary n-simplex


    Parameters
    ----------
    simplex : np.ndarray
        The simplex to sample from. The simplex is encoded as a 2d numpy array
        who's rows are the vertices of the simplex.
    num_samples : int
        Number of points to sample.
    closed : bool
        If true, the region sampled includes the boundary of the simplex.
    seed : Seed
        The seed for the random number generator.

    Returns
    -------
    `np.array` of shape (num_samples, dim+1), with the rows corresponding to
    individual samples.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be > 0")

    dim = simplex.shape[0] - 1
    # np.linalg.matrix_rank()
    barycentric = unit_simplex(dim, num_samples, closed, seed)
    return np.matmul(simplex.T, barycentric[:, :, np.newaxis]).squeeze()
