"""
Methods for generating synthetic topological data
"""

import numpy as np

from kaiju_mapper.types import Seed

Technically this is overly restrictive at the moment. I could not figure out
how to type hint `np.array_like[int]`. I doubt anyone will notice.
"""


def sphere(
    dim: int,
    radius: float = 1,
    num_samples: int = 1,
    seed: Seed | None = None,
) -> np.ndarray:
    r"""Sample uniformly from an n-sphere embedded in :math:`\mathbb{R}^{n+1}`

    This uses the algorithm described in Muller [1]_.

    Parameters
    ----------
    dim : np.ndarray
        The dimension of the sphere to sample from.
    radius : float
        The radius of the sphere to sample from.
    num_samples : int
        Number of points to sample.
    seed : Seed | None
        An optional seed for the random number generator.

    Returns
    -------
    `np.array` of shape (num_samples, dim+1), with the rows corresponding to
    individual samples.

    References
    ----------
    .. [1] Mervin E. Muller, "A Note on a Method for Generating Points Uniformly
           on N-Dimensional Spheres" 1959.

    Examples
    --------
    >>> import numpy as np
    >>> data = sphere(dim=2, radius=5, num_samples=4)
    >>> data.shape
    (4, 3)
    >>> np.linalg.norm(data, axis=1)
    array([5., 5., 5., 5.])
    """
    if num_samples < 1:
        raise ValueError("num_samples must be > 0")

    if dim <= 0:
        raise ValueError("dim must be at least 1")

    rng = np.random.default_rng(seed)

    result = rng.normal(size=(num_samples, dim + 1))
    result /= np.linalg.norm(result, axis=1)[:, np.newaxis]
    result *= radius
    return result
