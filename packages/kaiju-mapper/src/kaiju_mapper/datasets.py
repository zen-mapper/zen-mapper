"""
Methods for generating synthetic topological data
"""

import numpy as np
import numpy.typing as npt

from kaiju_mapper.types import Seed


def sphere(
    dim: int,
    radius: float | npt.ArrayLike = 1,
    num_samples: int = 1,
    seed: Seed | None = None,
) -> np.ndarray:
    r"""Sample uniformly from an n-sphere embedded in :math:`\mathbb{R}^{n+1}`

    This uses the algorithm described in Muller [1]_.

    Parameters
    ----------
    dim : np.ndarray
        The dimension of the sphere to sample from.
    radius : float | ArrayLike
        The radius of the sphere to sample from. If multiple radii are passed
        then data will be sampled uniformly from circles of each radius.
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

    >>> import numpy as np
    >>> data = sphere(dim=2, radius=[1, 2], num_samples=5, seed=42)
    >>> data.shape
    (5, 3)
    >>> np.linalg.norm(data, axis=1)
    array([2., 2., 2., 2., 1.])
    """
    if num_samples < 1:
        raise ValueError("num_samples must be > 0")

    if dim <= 0:
        raise ValueError("dim must be at least 1")

    rng = np.random.default_rng(seed)

    radius = np.array(radius)

    if radius.ndim > 1:
        raise ValueError

    if radius.ndim == 1:
        probabilities = np.sqrt(radius, dtype=float)
        probabilities /= np.sum(probabilities, dtype=float)
        ind = rng.choice(len(radius), size=num_samples, p=probabilities)
        radius = radius[ind].reshape(-1, 1)

    result = rng.normal(size=(num_samples, dim + 1))
    result /= np.linalg.norm(result, axis=1)[:, np.newaxis]
    result *= radius
    return result
