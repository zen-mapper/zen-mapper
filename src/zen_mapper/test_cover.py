from functools import reduce

import numpy as np
from hypothesis import given
from hypothesis.extra import numpy
from hypothesis.strategies import fixed_dictionaries, floats, integers, just

from zen_mapper.cover import Width_Balanced_Cover, _grid, rectangular_cover


def test_1d_rectangle():
    centers = np.array([3])
    data = np.array([0, 1, 2, 3, 4, 5, 6])

    widths = np.array([1])
    cover = rectangular_cover(centers, widths, data)
    assert len(cover) == len(centers)
    assert cover[0] == np.array(3)

    widths = np.array([2.1])
    cover = rectangular_cover(centers, widths, data)
    assert len(cover) == len(centers)
    assert set(cover[0]) == {2, 3, 4}


def test_2d_rectangle():
    data = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 1],
            [1, 2],
        ]
    )
    widths = np.array([2.1, 4.1])
    centers = np.array([[0, 0]])
    cover = rectangular_cover(centers, widths, data)
    assert len(cover) == len(centers)
    assert set(cover[0]) == {0, 1, 3}


@given(
    integers(min_value=1, max_value=6).flatmap(
        lambda n: fixed_dictionaries(
            {
                "start": numpy.arrays(int, (n,)),
                "stop": numpy.arrays(int, (n,)),
                "steps": integers(min_value=1, max_value=10),
                "dim": just(n),
            }
        )
    )
)
def test_grid_int_steps(args):
    start, stop, steps, dim = args["start"], args["stop"], args["steps"], args["dim"]
    grid = _grid(start, stop, steps)
    assert grid.shape[0] == steps**dim
    assert grid.shape[1] == dim
    assert np.all(start == grid[0])
    if steps == 1:
        assert np.all(start == grid[-1])
    else:
        assert np.all(stop == grid[-1])


@given(
    integers(min_value=1, max_value=6)
    .flatmap(
        lambda n: fixed_dictionaries(
            {
                "start": numpy.arrays(int, (n,)),
                "stop": numpy.arrays(int, (n,)),
                "steps": numpy.arrays(
                    int, (n,), elements={"min_value": 1, "max_value": 10}
                ),
                "dim": just(n),
            }
        )
    )
    .filter(lambda d: np.all(d["start"] != d["stop"]))
)
def test_grid_array_steps(args):
    start, stop, steps, dim = args["start"], args["stop"], args["steps"], args["dim"]
    grid = _grid(start, stop, steps)
    assert grid.shape[0] == np.prod(steps)
    assert grid.shape[1] == dim
    assert np.all(start == grid[0])
    assert np.all(stop[steps > 1] == grid[-1][steps > 1])
    assert np.all(start[steps == 1] == grid[-1][steps == 1])


@given(
    integers(min_value=1, max_value=1000).flatmap(
        lambda n: numpy.arrays(
            float,
            (n,),
            elements={
                "min_value": -100,
                "max_value": 100,
            },
        )
    ),
    integers(min_value=1, max_value=10),
    floats(min_value=0.1, max_value=0.5),
)
def test_width_balanced(data, n, gain):
    """Ensure that a width balanced cover covers the entire dataset"""
    cover_scheme = Width_Balanced_Cover(n, gain)
    covered_data = reduce(lambda acc, new: acc.union(new), cover_scheme(data), set())
    assert len(data) == len(covered_data)


def test_width_balanced_multiple_widths():
    """Make sure you can compute a width balanced cover with multiple widths defined"""
    data = np.arange(100).reshape((25, 4))
    gain = 0.4
    n = [1, 1, 2, 2]
    cover_scheme = Width_Balanced_Cover(n, gain)
    covered_data = reduce(lambda acc, new: acc.union(new), cover_scheme(data), set())
    assert len(data) == len(covered_data)


def test_width_balanced_int():
    """Ensure that width balanced covers handle integer data gracefully"""
    data = np.arange(100, dtype=int)
    cover_scheme = Width_Balanced_Cover(3, 0.4)
    covered_data = reduce(lambda acc, new: acc.union(new), cover_scheme(data), set())
    assert len(data) == len(covered_data)
