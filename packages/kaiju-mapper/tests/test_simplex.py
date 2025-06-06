import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from kaiju_mapper.datasets import simplex

dimension = st.integers(min_value=0, max_value=10)
num_samples = st.integers(min_value=1, max_value=1000)
scale = st.floats(min_value=0.5, max_value=3.0)


def unit_simplex(dimension: int) -> np.ndarray:
    return np.identity(dimension + 1, dtype=float)


@pytest.mark.parametrize("closed", [True, False])
@given(dimension=dimension, num_samples=num_samples)
def test_right_dimensions(dimension, num_samples, closed):
    data = simplex(
        simplex=unit_simplex(dimension),
        num_samples=num_samples,
        closed=closed,
    )

    assert data.shape == (num_samples, dimension + 1)


@pytest.mark.parametrize("closed", [True, False])
@given(dimension=dimension, num_samples=num_samples)
def test_in_simplex(dimension, num_samples, closed):
    data = simplex(
        simplex=unit_simplex(dimension),
        num_samples=num_samples,
        closed=closed,
    )

    assert_allclose(np.sum(data, axis=1), 1)


@pytest.mark.parametrize("closed", [True, False])
@given(dimension=dimension, scale=scale)
def test_mean(dimension, scale, closed):
    data = simplex(
        simplex=scale * unit_simplex(dimension),
        num_samples=10_000,
        closed=closed,
    )

    assert_allclose(np.mean(data, axis=0), scale / (dimension + 1), rtol=0.05)
