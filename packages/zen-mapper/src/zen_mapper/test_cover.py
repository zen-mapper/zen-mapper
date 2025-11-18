from functools import reduce

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra import numpy
from hypothesis.strategies import fixed_dictionaries, floats, integers, just

from zen_mapper.cover import (
    Data_Balanced_Cover,
    Width_Balanced_Cover,
    _grid,
    rectangular_cover,
)


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


def test_data_balanced_simple():
    """
    Test that Data_Balanced_Cover partitions data as expected for a tiny array.
    """
    cover = Data_Balanced_Cover(2, 0.5)
    data = np.array([10, 20, 30, 40, 50, 60])
    expected = {
        frozenset({0, 1, 2, 3}),
        frozenset({2, 3, 4, 5}),
    }

    groups = set(map(frozenset, cover(data)))
    assert expected == groups


def test_data_balanced_complete_coverage_random():
    """Ensure that for arbitrary random data all indices are covered."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=200)
    n_elements = 5
    overlap = 0.3

    cover = Data_Balanced_Cover(n_elements, overlap)
    groups = cover(data)

    all_indices = set(np.concatenate(groups))
    assert all_indices == set(range(len(data)))


def test_data_balanced_small_dataset_error():
    """Should raise ValueError if data has fewer points than groups."""
    data = np.arange(3)
    with pytest.raises(ValueError):
        Data_Balanced_Cover(5, 0.4)(data)


@pytest.mark.parametrize("overlap", [0, 1, -0.1, 2.0])
def test_data_balanced_invalid_overlap(overlap):
    """Should reject invalid percent_overlap."""
    with pytest.raises(ValueError):
        Data_Balanced_Cover(3, overlap)


def test_data_balanced_invalid_n_elements():
    """Should reject n_elements < 1."""
    with pytest.raises(ValueError):
        Data_Balanced_Cover(0, 0.4)


def test_data_balanced_only_1d_allowed():
    """Multi-dimensional data should be rejected."""
    data = np.random.rand(10, 2)
    with pytest.raises(ValueError):
        Data_Balanced_Cover(3, 0.4)(data)


def test_data_balanced_cover_structure():
    """Check approximate interval structure and order consistency."""
    data = np.linspace(0, 100, 101)
    n_elements = 4
    overlap = 0.25
    cover = Data_Balanced_Cover(n_elements, overlap)
    groups = cover(data)

    # No repeat elements in any cover element
    for g in groups:
        assert np.unique_values(g).size == g.size

    # Pairwise overlap is consistent
    for g1, g2 in zip(groups, groups[1:]):
        assert np.intersect1d(g1, g2).size == 7

    all_idx = np.concatenate(groups)

    # Every id is present in the original data
    assert np.all(all_idx < len(data))

    # Every index is present
    assert set(all_idx) == set(range(len(data)))
