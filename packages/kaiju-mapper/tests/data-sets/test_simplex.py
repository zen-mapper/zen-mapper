import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from kaiju_mapper.datasets import unit_simplex

dimension = st.integers(min_value=0, max_value=100)
num_samples = st.integers(min_value=1, max_value=1000)


@pytest.mark.parametrize("closed", [True, False])
@given(dimension=dimension, num_samples=num_samples)
def test_right_dimensions(dimension, num_samples, closed):
    data = unit_simplex(dim=dimension, num_samples=num_samples, closed=closed)

    assert data.shape == (num_samples, dimension + 1)


@pytest.mark.parametrize("closed", [True, False])
@given(dimension=dimension, num_samples=num_samples)
def test_in_simplex(dimension, num_samples, closed):
    data = unit_simplex(dim=dimension, num_samples=num_samples, closed=closed)

    assert_allclose(np.sum(data, axis=1), 1)


@pytest.mark.parametrize("closed", [True, False])
@given(dimension=dimension)
def test_mean(dimension, closed):
    data = unit_simplex(dim=dimension, num_samples=10_000, closed=closed)

    assert_allclose(np.mean(data, axis=0), 1 / (dimension + 1), rtol=0.05)
