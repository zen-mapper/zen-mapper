from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import numpy.typing as npt

__all__ = [
    "rectangular_cover",
    "width_balanced_cover",
    "data_balanced_cover",
]

logger = logging.getLogger("zen_mapper")


def rectangular_cover(
    centers: np.ndarray,
    widths: np.ndarray,
    data: np.ndarray,
    tol: float = 1e-9,
) -> list[np.ndarray]:
    """Partition data points into multi-dimensional rectangular cover elements.

    Note:
        This is a low-level structural function that requires pre-computed
        bounding box geometries. For most use cases, you should use
        :func:`width_balanced_cover` or :func:`data_balanced_cover` instead,
        which handle the geometry generation automatically.

    Args:
        centers: The coordinates of the centers for each hyper-rectangle. Shape
            should be `(n_centers, n_features)` or `(n_centers,)`.
        widths: The width of the covering elements. Must be broadcastable
            against the feature dimensions (e.g., a scalar or a 1D array of
            shape `(n_features,)`).
        data: The dataset to be partitioned into the cover elements. Shape
            should be `(n_samples, n_features)` or `(n_samples,)`.
        tol: A small numerical tolerance added to the boundary calculations to
            prevent floating-point precision issues for points resting exactly
            on an edge. Defaults to 1e-9.

    Returns:
        A list of length `n_centers`. Each entry is a 1D array of integers
        containing the row indices of `data` that fall inside that specific
        rectangular cover element.

    See Also:
        :func:`width_balanced_cover`: compute a cover comprised of equally sized
        rectangular elements.

        :func:`data_balanced_cover` compute a cover where each element has the
        same number of data points.
    """
    if centers.ndim == 1:
        centers = centers.reshape(-1, 1)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    distances = np.abs(data - centers[:, None])
    in_bounds = np.all(
        distances <= (widths + tol) / 2,
        axis=2,
    )
    return [np.flatnonzero(mask) for mask in in_bounds]


def _grid(
    start: npt.ArrayLike,
    stop: npt.ArrayLike,
    steps: npt.ArrayLike,
) -> np.ndarray:
    """Create a flat coordinate grid from a set of N-dimensional bounds.

    Args:
        start: The starting coordinates for each dimension.
        stop: The ending coordinates for each dimension, inclusive.
        steps: The number of grid points sampled along each dimension. If an
            integer, the same number of steps is taken acrosss all dimensions.

    Returns:
        An array where each row represents a unique coordinate point in the grid.

    Raises:
        ValueError: If `start`, `stop`, and `steps` cannot be broadcast to a
            common shape
        ValueError: If `steps` has non-integral dtype

    Examples:
        >>> _grid(start=[0,10], stop=[1, 20], steps=2)
        array([[ 0., 10.],
               [ 1., 10.],
               [ 0., 20.],
               [ 1., 20.]])

        >>> _grid(start = [0,10], stop=[1,20], steps=[2,3])
        array([[ 0., 10.],
               [ 1., 10.],
               [ 0., 15.],
               [ 1., 15.],
               [ 0., 20.],
               [ 1., 20.]])
    """
    start, stop = np.atleast_1d(start), np.atleast_1d(stop)
    steps = np.asarray(steps)

    if not np.issubdtype(steps.dtype, int):
        raise ValueError("Steps must have an integral type")

    dims = (
        np.linspace(begin, end, num=num)
        for begin, end, num in np.broadcast(start, stop, steps)
    )

    grid = np.meshgrid(*dims)
    return np.stack(grid, axis=-1).reshape(-1, len(start))


class WidthBalancedMetadata(TypedDict):
    """Geometric information about a width balanced cover

    After fitting a cover using :func:`width_balanced_cover` this contains all
    the information you would need to reconstruct the cover using
    :func:`rectangular_cover`, fit new data, or visualize the cover
    geometrically.
    """

    centers: np.ndarray
    """The coordinates of the centers for each rectangular covering element."""
    widths: np.ndarray
    """The calculated width of the covering elements across each dimension."""


def width_balanced_cover(
    n_elements: npt.ArrayLike,
    percent_overlap: float,
    data: npt.ArrayLike,
) -> tuple[list[np.ndarray], WidthBalancedMetadata]:
    """Compute a cover of equally sized rectangular elements.

    The widths of the intervals are calculated so that the entire range of the
    data is spanned by the desired number of elements with the specified
    overlap percentage.

    Args:
        n_elements: The number of covering elements along each dimension. If
            the data is dimension :math:`d` and this is a scalar :math:`n` this
            results in :math:`n^d` covering elements.
        percent_overlap: A number between 0 and 1 representing the amount of
            overlap between adjacent covering elements.
        data: The input data array to be covered. Of shape `(n_samples,
            n_features)` or `(n_samples,)`.

    Returns:
        A tuple `(cover, metadata)` where `cover` is the fitted cover and `metadata`
        is a dictionary containing geometric properties of the generated cover.
        See :class:`WidthBalancedMetadata` for more information.

    Raises:
        ValueError: If any value in `n_elements` is  less than 1.
        ValueError: If `percent_overlap` is not in the open interval (0,1)
    """

    n_elements = np.atleast_1d(n_elements)

    data = np.asarray(data, dtype=float)

    if data.ndim < 2:
        logger.warning(
            "Data has shape %s, reshaping to (%s, 1)",
            data.shape,
            data.size,
        )
        data = data.reshape(-1, 1)

    if data.ndim > 2:
        raise ValueError(
            f"Shape of data must be (n_samples, n_features), got f{data.shape}"
        )

    if np.any(n_elements < 1):
        raise ValueError(f"n_elements must be at least 1, got {n_elements}")

    if not 0 < percent_overlap < 1:
        raise ValueError(
            f"percent_overlap must be in the range (0,1), got {percent_overlap}"
        )

    logger.info("Computing the width balanced cover")

    upper_bound = np.max(data, axis=0)
    lower_bound = np.min(data, axis=0)

    width = (upper_bound - lower_bound) / (
        n_elements - (n_elements - 1) * percent_overlap
    )
    width = width.flatten()

    # Compute the centers of the "lower left" and "upper right" cover
    # elements
    upper_bound -= width / 2
    lower_bound += width / 2

    centers = _grid(lower_bound, upper_bound, n_elements)
    return rectangular_cover(centers, width, data), {
        "centers": centers,
        "widths": width,
    }


def data_balanced_cover(
    n_elements: int,
    percent_overlap: float,
    data: npt.ArrayLike,
) -> tuple[list[np.ndarray], DataBalancedMetadata]:
    r"""Compute a cover with roughly equal data points per interval

    The cover is constructed by partitioning the sorted indices :math:`[0, \dots, N-1]`
    into intervals of approximately equal size, then mapping those
    index-regions back to the original data positions.

    Each bin has a base size and step calculated as:

    .. math::
        base\_size = \frac{N}{k - (k - 1) \times \text{overlap}}

    .. math::
        step = base\_size \times (1 - \text{overlap})

    where :math:`k` is `n_elements`.

    Args:
        n_elements: The number of intervals (cover elements) to create. Must be
            at least 1.
        percent_overlap: The fractional overlap between adjacent intervals.
            Must be between 0 and 1 exclusive.

    Returns:
        A tuple `(cover, metadata)` where `cover` is the fitted cover and
        `metadata` is a dictionary containing geometric properties of the
        generated cover. See :class:`DataBalancedMetadata` for more
        information.

    Raises:
        ValueError: If `n_elements` < 1 or `percent_overlap` is not in the
            open range (0, 1).

    Note:
        A `percent_overlap` of 0.5 means each interval shares approximately 50%
        of its points with the subsequent interval.

    Examples:
        >>> data = np.array([10, 11, 12, 40, 55, 60])
        >>> cover, meta = data_balanced_cover(2, 0.5, data)
        >>> cover
        [array([0, 1, 2, 3]), array([2, 3, 4, 5])]
        >>> [ data[e] for e in cover ]
        [array([10, 11, 12, 40]), array([12, 40, 55, 60])]
        >>> bounds = meta["bounds"]
        >>> bounds
        array([[0, 3],
               [2, 5]])
        >>> data[bounds]
        array([[10, 40],
               [12, 60]])
    """

    data = np.atleast_1d(data)

    if data.ndim != 1:
        raise ValueError(
            f"Data_Balanced_Cover only supports 1-dimensional input"
            f"data but received data with shape: {data.shape}"
        )

    n = len(data)

    if n < n_elements:
        raise ValueError("Number of data points must be >= n_elements")

    logger.info("Computing the data balanced cover")

    sort_idx = np.argsort(data)
    idxs = np.arange(n)
    cover_idxs, _ = width_balanced_cover(
        n_elements=n_elements,
        percent_overlap=percent_overlap,
        data=idxs,
    )

    cover = [sort_idx[g] for g in cover_idxs]

    bounds = np.fromiter(
        ((e[0], e[-1]) for e in cover),
        dtype=np.dtype((int, 2)),
        count=len(cover),
    )

    return cover, {"bounds": bounds}


class DataBalancedMetadata(TypedDict):
    bounds: np.ndarray
    """The indices of the bounds for each interval. Shape `(num_intervals, 2)`"""
