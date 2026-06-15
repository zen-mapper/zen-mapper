import logging

import numpy as np
import numpy.typing as npt

from .types import Cover, CoverScheme

__all__ = [
    "precomputed_cover",
    "rectangular_cover",
    "Width_Balanced_Cover",
    "Data_Balanced_Cover",
]

logger = logging.getLogger("zen_mapper")


def precomputed_cover(cover: Cover) -> CoverScheme:
    """A precomputed cover

    Parameters
    ----------
    cover : Cover
        the precomputed cover to use
    """

    def inner(*_):
        return cover

    return inner  # type: ignore


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


class Width_Balanced_Cover:
    """A cover comprised of equally sized rectangular elements

    Parameters
    ----------
    n_elements : ArrayLike
        The number of covering elements along each dimension. If the data is
        dimension $d$ and this is a scalar $n$ this results in $n^d$ covering elements.
    percent_overlap : float
        a number between 0 and 1 representing the ammount of overlap between
        adjacent covering elements.


    Raises
    ------
    Value Error
        if n_elements < 1
    Value Error
        if percent_overlap is not in (0,1)
    """

    def __init__(self, n_elements: npt.ArrayLike, percent_overlap: float):
        n_elements = np.array([n_elements], dtype=int)

        if np.any(n_elements < 1):
            raise ValueError("n_elements must be at least 1")

        if not 0 < percent_overlap < 1:
            raise ValueError("percent_overlap must be in the range (0,1)")

        self.n_elements = n_elements
        self.percent_overlap = percent_overlap

    def __call__(self, data):
        logger.info("Computing the width balanced cover")

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        upper_bound = np.max(data, axis=0).astype(float)
        lower_bound = np.min(data, axis=0).astype(float)

        width = (upper_bound - lower_bound) / (
            self.n_elements - (self.n_elements - 1) * self.percent_overlap
        )
        width = width.flatten()
        self.width = width

        # Compute the centers of the "lower left" and "upper right" cover
        # elements
        upper_bound -= width / 2
        lower_bound += width / 2

        centers = _grid(lower_bound, upper_bound, self.n_elements)
        self.centers = centers
        return rectangular_cover(centers, width, data)


class Data_Balanced_Cover:
    r"""
    A cover of 1D data with roughly equal data points per interval.

    The cover is constructed by partitioning the sorted indices :math:`[0, \dots, N-1]`
    into intervals of approximately equal size, then mapping those
    index-regions back to the original data positions.

    Each bin has a base size and step calculated as:

    .. math::
        base\_size = \frac{N}{k - (k - 1) \times \text{overlap}}

    .. math::
        step = base\_size \times (1 - \text{overlap})

    where :math:`k` is `n_elements`.

    Parameters
    ----------
    n_elements : int
        The number of intervals (cover elements) to create. Must be :math:`\ge 1`.
    percent_overlap : float
        The fractional overlap between adjacent intervals, :math:`0 <
        \text{overlap} < 1`.

    Attributes
    ----------
    n_elements : int
        The number of cover elements.
    percent_overlap : float
        The fractional overlap.

    Raises
    ------
    ValueError
        If `n_elements` < 1 or `percent_overlap` is not in the range (0, 1).

    Notes
    -----
    A `percent_overlap` of 0.5 means each interval shares approximately 50%
    of its points with the subsequent interval.
    """

    def __init__(self, n_elements: int, percent_overlap: float):
        self._cover = Width_Balanced_Cover(
            n_elements=n_elements,
            percent_overlap=percent_overlap,
        )
        self.n_elements = n_elements
        self.percent_overlap = percent_overlap

    def __call__(self, data: npt.ArrayLike):
        """
        Partition the input data into overlapping intervals containing
        approximately equal numbers of points.

        This method sorts the input data and applies a width-balanced cover
        to the indices. It then maps these index-based regions back to the
        original data indices to create the balanced cover.

        Parameters
        ----------
        data : array_like
            A 1-dimensional array of data points to be partitioned.

        Returns
        -------
        list of ndarray
            A list containing the indices of the original data points
            belonging to each cover element. Each element in the list is
            an `np.ndarray`.

        Raises
        ------
        ValueError
            If the input `data` is not 1-dimensional.
        ValueError
            If the number of points in `data` is less than the requested
            `n_elements`.
        """
        data = np.asarray(data, dtype=float)

        if data.ndim != 1:
            raise ValueError(
                f"Data_Balanced_Cover only supports 1-dimensional input"
                f"(projected) data but received data with dim: {data.ndim}"
            )

        logger.info("Computing the data balanced cover")

        n = len(data)

        if n < self.n_elements:
            raise ValueError("Number of data points must be >= n_elements")

        sort_idx = np.argsort(data)
        idxs = np.arange(n)
        cover_idxs = self._cover(idxs)

        cover = [sort_idx[g] for g in cover_idxs]

        return cover
