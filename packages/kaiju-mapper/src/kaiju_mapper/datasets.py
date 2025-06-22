"""
Methods for generating synthetic topological data
"""

import sys

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
    simplex: npt.ArrayLike,
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

    _simplex = np.asarray(simplex)
    dim = _simplex.shape[0] - 1
    barycentric = unit_simplex(
        dim=dim,
        num_samples=num_samples,
        closed=closed,
        seed=seed,
    )
    return np.matmul(barycentric, _simplex)


def ball(
    dim: int,
    radius: float = 1,
    num_samples: int = 1,
    seed: Seed = None,
) -> np.ndarray:
    """Sample uniformly from an n-ball


    Parameters
    ----------
    dim : np.ndarray
        The dimension of the ball to sample from.
    radius : float
        The radius of the ball to sample from
    num_samples : int
        Number of points to sample.
    seed : Seed
        The seed for the random number generator.

    Returns
    -------
    `np.array` of shape (num_samples, dim+1), with the rows corresponding to
    individual samples.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be greater than 0")

    if dim <= 0:
        raise ValueError("dim must be at least 1")

    if radius <= 0:
        raise ValueError("radius must be greater than 0")

    rng = np.random.default_rng(seed)

    directions = sphere(dim, num_samples=num_samples, seed=rng)
    radii = radius * np.sqrt(rng.uniform(0, 1, size=num_samples))
    return directions * radii[:, np.newaxis]


def annulus(
    minor_radius: float,
    major_radius: float,
    dim: int = 1,
    num_samples: int = 1,
    seed: Seed = None,
) -> np.ndarray:
    """Sample uniformly from an annulus


    Parameters
    ----------
    minor_radius: float
        The radius of the inside of the annulus
    major_radius: float
        The radius of the outside of the annulus
    dim : np.ndarray
        The dimension of the annulus to sample from.
    num_samples : int
        Number of points to sample.
    seed : Seed
        The seed for the random number generator.

    Returns
    -------
    `np.array` of shape (num_samples, dim+1), with the rows corresponding to
    individual samples.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be > 0")

    if dim <= 0:
        raise ValueError("dim must be at least 1")

    if minor_radius <= 0:
        raise ValueError("minor_radius must be greater than 0")

    if major_radius <= 0:
        raise ValueError("major_radius must be greater than 0")

    if minor_radius >= major_radius:
        raise ValueError("major_radius must be greater than minor_radius")

    rng = np.random.default_rng(seed)

    directions = sphere(dim, num_samples=num_samples, seed=rng)
    radii = (
        np.sqrt(rng.uniform(0, 1, size=num_samples)) * (major_radius - minor_radius)
        + minor_radius
    )
    return directions * radii[:, np.newaxis]


def flat_torus(
    dim: int,
    num_samples: int = 1,
    seed: Seed = None,
) -> np.ndarray:
    if dim <= 0:
        raise ValueError("dim must be at least 1")

    result = sphere(dim=1, radius=1, num_samples=dim * num_samples, seed=seed)
    return result.reshape(-1, 2 * dim)
