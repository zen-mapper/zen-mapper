import logging
import sys
from collections.abc import Iterator
from typing import Protocol

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


import numpy as np
import numpy.typing as npt

logger = logging.getLogger("zen_mapper")


class Cover(Protocol):
    def __len__(self: Self) -> int: ...

    def __iter__(self: Self) -> Iterator[np.ndarray]: ...


class CoverScheme(Protocol):
    def __call__(self: Self, data: np.ndarray) -> Cover: ...


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
