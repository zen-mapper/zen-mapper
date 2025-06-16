import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from kaiju_mapper.datasets import ball

dimension = st.integers(min_value=1, max_value=100)
num_samples = st.integers(min_value=1, max_value=1000)
radius = st.floats(min_value=0.1, max_value=1e20, allow_infinity=False, allow_nan=False)


@given(dimension=dimension, num_samples=num_samples, radius=radius)
def test_right_dimensions(dimension, num_samples, radius):
    data = ball(dim=dimension, num_samples=num_samples, radius=radius)

    assert data.shape == (num_samples, dimension + 1)


@given(dimension=dimension, num_samples=num_samples, radius=radius)
def test_on_sphere(dimension, num_samples, radius):
    data = ball(dim=dimension, num_samples=num_samples, radius=radius)

    assert np.all(np.linalg.norm(data, axis=1) <= radius)


@given(dimension=dimension)
def test_mean(dimension):
    data = ball(dim=dimension, num_samples=10_000)

    assert_allclose(np.mean(data), 0, atol=0.015)
