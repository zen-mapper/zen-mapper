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


def rectangular_cover(centers, widths, data, tol=1e-9):
    if len(centers.shape) == 1:
        centers = centers.reshape(-1, 1)

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    distances = np.abs(data - centers[:, None])
    return list(
        map(
            np.flatnonzero,
            np.all(
                distances * 2 - widths <= tol,
                axis=2,
            ),
        )
    )


def _grid(start, stop, steps):
    """Create an n-dimensional grid from start to stop with steps

    Parameters
    ----------
    start : ndarray
        The point to start at
    stop : ndarray
        The point to stop at
    steps : int | ndarray
        The number of grid points for each direction

    Raises
    ------

    ValueError
        If len(start) != len(stop)
    """

    if len(start) != len(stop):
        raise ValueError("Start and stop points need to have same dimension")

    dims = (
        np.linspace(begin, end, num=num)
        for begin, end, num in np.broadcast(start, stop, steps)
    )
    grid = np.meshgrid(*dims)
    return np.column_stack([dim.reshape(-1) for dim in grid])


class Width_Balanced_Cover:
    """A cover comprised of equally sized rectangular elements

    Parameters
    ----------
    n_elements : ArrayLike
        the number of covering elements along each dimension. If the data is
        dimension d this results in d^n covering elements.

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
        The number of intervals (cover elements) to create. Must be $\ge 1$.
    percent_overlap : float
        The fractional overlap between adjacent intervals, $0 < \text{overlap} < 1$.

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
