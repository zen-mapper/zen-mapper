"""
Methods for generating synthetic topological data
"""

import numpy as np
import numpy.typing as npt

from kaiju_mapper.types import Seed


def sphere(
    dim: int,
    radius: float | None = None,
    center: npt.ArrayLike | None = None,
    num_samples: int = 1,
    seed: Seed | None = None,
) -> np.ndarray:
    r"""Sample uniformly from an n-sphere embedded in :math:`\mathbb{R}^{n+1}`

    This uses the algorithm described in Muller [1]_.

    Parameters
    ----------
    dim : np.ndarray
        The dimension of the sphere to sample from.
    radius : float | None
        The radius of the sphere to sample from. Defaults to 1.
    center : ArrayLike | None
        The center of the sphere to sample from. Defaults to the origin.
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
    >>> data = sphere(dim=2, radius=2, center=(0, 1, 0), num_samples=4)
    >>> data.shape
    (4, 3)
    >>> np.linalg.norm(data - (0, 1, 0), axis=1)
    array([2., 2., 2., 2.])
    """
    # Initialize variables

    rng = np.random.default_rng(seed)

    if center is not None:
        center = np.asarray(center)

    # Validate inputs

    if num_samples < 1:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    if dim <= 0:
        raise ValueError(f"dim must be at least 1, got {dim}")

    if radius is not None and radius < 0:
        raise ValueError(f"radius must be positive, got {radius}")

    if center is not None and (center.ndim != 1 or len(center) != dim + 1):
        raise ValueError(f"center must have shape ({dim + 1},), got {center.shape}")

    # Sample from the circles
    result = rng.normal(size=(num_samples, dim + 1))
    result /= np.linalg.norm(result, axis=1)[:, np.newaxis]

    if radius is not None:
        result *= radius

    if center is not None:
        result += center

    return result
