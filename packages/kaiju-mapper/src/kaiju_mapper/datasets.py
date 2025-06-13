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
    radius: float = 1,
    num_samples: int = 1,
    seed: Seed = None,
) -> np.ndarray:
    """Sample uniformly from an n-sphere

    This uses the algorithm described in Muller _[1].

    Parameters
    ----------
    dim : np.ndarray
        The dimension of the sphere to sample from.
    radius : float
        The radius of the sphere to sample from.
    num_samples : int
        Number of points to sample.
    seed : Seed
        The seed for the random number generator.

    Returns
    -------
    `np.array` of shape (num_samples, dim+1), with the rows corresponding to
    individual samples.

    .. [1] Mervin E. Muller, "A Note on a Method for Generating Points
    Uniformly on N-Dimensional Spheres" 1959.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be > 0")

    if dim <= 0:
        raise ValueError("dim must be at least 1")

    rng = np.random.default_rng(seed)

    result = rng.normal(size=(num_samples, dim + 1))
    result /= np.linalg.norm(result, axis=1)[:, np.newaxis]
    return radius * result


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
