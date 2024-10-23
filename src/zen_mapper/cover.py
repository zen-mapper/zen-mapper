import logging
from collections.abc import Iterator
from typing import Protocol, Self
import numpy as np

# Initialize logger for zen_mapper
logger = logging.getLogger("zen_mapper")

# Define a Protocol for Cover, which specifies the structure of objects that 
# behave as a "cover". This includes implementing the __len__ and __iter__ methods.
class Cover(Protocol):
    """Protocol representing a collection of covering elements"""

    def __len__(self: Self) -> int:
        """Returns the number of elements in the cover"""
        ...

    def __iter__(self: Self) -> Iterator[np.ndarray]:
        """Allows iteration over covering elements as ndarrays"""
        ...

# Define a Protocol for CoverScheme, which specifies objects that can be 
# applied to data to produce a Cover.
class CoverScheme(Protocol):
    """Protocol representing a scheme that produces a Cover when applied to data"""

    def __call__(self: Self, data: np.ndarray) -> Cover:
        """Applies the cover scheme to the provided data to generate a cover"""
        ...


def rectangular_cover(centers, widths, data, tol=1e-9):
    """Generate a rectangular cover for the data using specified centers and widths

    Parameters
    ----------
    centers : ndarray
        The center points of the covering elements.
    
    widths : ndarray
        The widths of the covering elements along each dimension.
    
    data : ndarray
        The data points to be covered.
    
    tol : float, optional
        A tolerance for numeric errors in comparison, by default 1e-9

    Returns
    -------
    list of ndarray
        A list of arrays, each representing the indices of data points 
        belonging to the respective covering element.

    Notes
    -----
    Generates covering elements in each dimension and checks 
    which points from `data` fall within the bounds of each covering element.
    """
    
    # Ensure centers and data are 2D arrays
    if len(centers.shape) == 1:
        centers = centers.reshape(-1, 1)

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # Compute distances between data points and cover centers
    distances = np.abs(data - centers[:, None])
    
    # Return a list of indices of data points that fall within the cover elements
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
    """Creates an n-dimensional grid from start to stop with steps

    Parameters
    ----------
    start : ndarray
        The start points for each dimension.
    
    stop : ndarray
        The stop points for each dimension.
    
    steps : int | ndarray
        The number of steps (grid points) for each dimension.

    Returns
    -------
    ndarray
        An n-dimensional grid of points as an ndarray.

    Raises
    ------
    ValueError
        If the lengths of `start` and `stop` arrays are not the same.

    Notes
    -----
    Creates a grid of points over a multi-dimensional space which is 
    evenly spaced between start and stop points along each dimension.
    """
    if len(start) != len(stop):
        raise ValueError("Start and stop points need to have the same dimension")

    # Create a grid by generating linear space between start and stop for each dimension
    dims = (
        np.linspace(begin, end, num=num)
        for begin, end, num in np.broadcast(start, stop, steps)
    )
    grid = np.meshgrid(*dims)
    
    # Reshape the grid points and return them as a single array
    return np.column_stack([dim.reshape(-1) for dim in grid])


class Width_Balanced_Cover:
    """A cover comprised of equally sized rectangular elements

    Parameters
    ----------
    n_elements : int
        The number of covering elements along each dimension. If the data is
        of dimension `d`, this results in `n_elements^d` covering elements.
    
    percent_overlap : float
        A number between 0 and 1 representing the amount of overlap between
        adjacent covering elements along each dimension.

    Raises
    ------
    ValueError
        If `n_elements` is less than 1.
    ValueError
        If `percent_overlap` is not in the range (0, 1).

    Notes
    -----
    Represents a covering scheme where covering elements are rectangular
    and equally spaced with a specified amount of overlap. The cover adapts to the
    data's bounding box and evenly distributes covering elements within that box.
    """

    def __init__(self, n_elements: int, percent_overlap: float):
        if n_elements < 1:
            raise ValueError("n_elements must be at least 1")

        if not 0 < percent_overlap < 1:
            raise ValueError("percent_overlap must be in the range (0,1)")

        self.n_elements = n_elements
        self.percent_overlap = percent_overlap

    def __call__(self, data):
        """Generate the width-balanced cover for the provided data.

        Parameters
        ----------
        data : ndarray
            The data points to be covered.

        Returns
        -------
        list of ndarray
            A list of arrays representing the covering elements, where each
            array contains the indices of the data points covered.

        Notes
        -----
        The function computes the bounding box of the data and generates equally
        spaced rectangular covering elements that overlap by the specified amount.
        """
        logger.info("Computing the width balanced cover")

        # Reshape data if it's 1D
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Compute the bounding box of the data
        upper_bound = np.max(data, axis=0).astype(float)
        lower_bound = np.min(data, axis=0).astype(float)

        # Calculate the width of each covering element
        width = (upper_bound - lower_bound) / (
            self.n_elements - (self.n_elements - 1) * self.percent_overlap
        )
        self.width = width

        # Adjust bounds to compute the centers of the covering elements
        upper_bound -= width / 2
        lower_bound += width / 2

        # Generate centers for the covering elements using a grid
        centers = _grid(lower_bound, upper_bound, self.n_elements)
        self.centers = centers

        # Return the rectangular cover based on the centers and widths
        return rectangular_cover(centers, width, data)
