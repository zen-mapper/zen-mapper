"""
Methods for generating synthetic topological data
"""

import sys
import numpy as np
import numpy.typing as npt

from kaiju_mapper.types import Seed


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
    Simplex" 2004.
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


def torus(
    minor_radius: float,
    major_radius: float,
    num_samples: int = 1,
    seed: Seed = None,
):
    """Sample uniformly from the torus embedded in ℝ³.

    Uses algorithm one from  Diaconis et al _[1].

    Parameters
    ----------
    minor_radius: float
        The minor radius of the torus. The width of the "tube".
    major_radisu: float:
        The major radius of the torus. The turning radius of the "tube".
    num_samples : int
        Number of points to sample.
    seed : Seed
        The seed for the random number generator.

    Returns
    -------
    `np.array` of shape (num_samples, 3), with the rows corresponding to
    individual samples.

    .. [1] Persi Diaconis, Susan Holmes, and Mehrad Shahshahani, "Sampling From
    a Manifold" 2012.
    """

    if minor_radius <= 0:
        raise ValueError(f"Minor radius must be ≥ 0, got {minor_radius}")

    if major_radius <= 0:
        raise ValueError(f"Major radius must be ≥ 0, got {major_radius}")

    if major_radius < minor_radius:
        raise ValueError(
            f"Major radius must be ≥ minor radius, got {major_radius=} {minor_radius=}"
        )

    if num_samples < 0:
        raise ValueError(f"Num samples must be positive, got {num_samples=}")

    rng = np.random.default_rng(seed)

    phi = rng.uniform(0, 2 * np.pi, num_samples)
    theta = np.full(num_samples, np.nan)
    wanted_samples = num_samples
    radius_ratio = major_radius / minor_radius

    while wanted_samples != 0:
        num_requested_samples = (
            2 * wanted_samples
        )  # There is a 50% chance of rejecting a sample
        x = rng.uniform(0, 2 * np.pi, num_requested_samples)
        y = rng.uniform(-radius_ratio, radius_ratio, num_requested_samples)
        kept_samples = x[y < np.cos(x)][:wanted_samples]
        index = num_samples - wanted_samples
        theta[index : index + len(kept_samples)] = kept_samples
        wanted_samples -= len(kept_samples)

    result = np.empty((num_samples, 3))
    result[:, 0] = (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
    result[:, 1] = (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
    result[:, 2] = minor_radius * np.sin(theta)
    return result


def klein_bottle(
    dim: int,
    num_samples: int = 1,
    scale: float = 1.0,
    seed: Seed = None,
) -> np.ndarray:
    """Sample uniformly from an n-dimensional Klein bottle.

    For dim=2, this generates the classic Klein bottle embedded in R^4.
    For higher dimensions, this generates a generalized Klein bottle as described
    in Davis [1]_.

    Parameters
    ----------
    dim : int
        The dimension of the Klein bottle manifold. Must be >= 2.
    num_samples : int
        Number of points to sample.
    scale : float
        The overall scale of the Klein bottle.
    seed : Seed
        The seed for the random number generator.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, 2*dim) containing points on the
        n-dimensional Klein bottle embedded in R^(2*dim).

    .. [1] Donald M. Davis, "n-dimensional Klein bottles", arXiv:1706.03704 [math.AT], 2017.

    Raises
    ------
    ValueError
        If dim < 2, num_samples < 1, or radius <= 0.

    Notes
    -----
    The Klein bottle is a non-orientable surface. For dim=2, this uses
    the standard parameterization embedded in R^4. For higher dimensions,
    this creates a product space of Klein bottles and circles following
    the construction in Davis [1]_.

    Examples
    --------
    >>> # Generate classic 2D Klein bottle in R^4
    >>> data = klein_bottle(dim=2, num_samples=100, seed=42)
    >>> data.shape
    (100, 4)

    >>> # Generate 3D Klein bottle in R^5
    >>> data = klein_bottle(dim=3, num_samples=50, scale=2.0)
    >>> data.shape
    (50, 5)
    """
    if dim < 2:
        raise ValueError("dim must be at least 2")

    if num_samples < 1:
        raise ValueError("num_samples must be > 0")

    if scale <= 0:
        raise ValueError("scale must be greater than 0")

    rng = np.random.default_rng(seed)

    # Generate parameters
    thetas = rng.uniform(0, 2 * np.pi, size=(num_samples, dim - 1))
    t = rng.uniform(0, np.pi, size=(num_samples,))

    alpha = np.zeros(shape=(num_samples, dim + 1))
    alpha[:, 0] = 5 * np.sin(t)
    alpha[:, 1] = 2 * (np.sin(t) ** 2) * np.cos(t)

    # alphaPrime = np.zeros((num_samples, dim + 1))
    # alphaPrime[:, 0] = 5 * np.cos(t)
    # alphaPrime[:, 1] = 4 * (np.cos(t) ** 2) * np.sin(t) - np.sin(t) ** 3

    alphaPrimeNorm = np.zeros((num_samples, dim + 1))
    alphaPrimeNorm[:, 0] = (5 * np.cos(t)) / (
        (5 * np.cos(t)) ** 2 + (4 * (np.cos(t) ** 2) * np.sin(t) - np.sin(t) ** 3) ** 2
    ) ** 0.5
    alphaPrimeNorm[:, 1] = (4 * (np.cos(t) ** 2) * np.sin(t) - np.sin(t) ** 3) / (
        (5 * np.cos(t)) ** 2 + (4 * (np.cos(t) ** 2) * np.sin(t) - np.sin(t) ** 3) ** 2
    ) ** 0.5

    j1 = -1 * alphaPrimeNorm[:, 1]
    j2 = alphaPrimeNorm[:, 0]

    s = 0.5

    r = np.zeros(dim - 1)
    for i in range(dim - 2):
        r[i] = 2 ** (dim - i + 1) / (2 ** (dim + 1) - 5)
    r[dim - 2] = s - 0.5 + 3 / (2 * (2 ** (dim + 1) - 5))

    w = np.zeros((num_samples, dim))
    for i in range(num_samples):
        w[i, dim - 1] = r[dim - 2]

    for i in reversed(range(1, dim - 1)):
        w[:, i] = r[i - 1] + w[:, i + 1] * np.cos(thetas[:, i])

    x = np.zeros((num_samples, dim))
    x[:, 0] = w[:, 1]
    for i in range(1, dim):
        x[:, i] = w[:, i] * np.sin(thetas[:, i - 1])

    # Called r(t) in the paper
    R = 0.5 - 2 * (2 * t - np.pi) * (t * (np.pi - t)) ** 0.5 / (
        (np.pi**2) * (2 ** (dim + 1) - 5)
    )

    result = np.zeros((num_samples, dim + 2))
    result[:, -1] = np.sin(2 * t)
    for i in range(2, dim + 1):
        result[:, i] = +x[:, i - 1] * R

    result[:, 0] = alpha[:, 0] + x[:, 0] * j1 * R
    result[:, 1] = alpha[:, 1] + x[:, 0] * j2 * R

    # scale
    result *= scale

    return result
