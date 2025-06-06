import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from kaiju_mapper.datasets import annulus

dimension = st.integers(min_value=1, max_value=100)
num_samples = st.integers(min_value=1, max_value=1000)


@st.composite
def radii(draw):
    minor_radius = draw(
        st.floats(min_value=0.1, max_value=8, allow_infinity=False, allow_nan=False)
    )
    major_radius = draw(
        st.floats(
            min_value=minor_radius + 0.1,
            exclude_min=True,
            max_value=16,
            allow_infinity=False,
            allow_nan=False,
        )
    )
    return (minor_radius, major_radius)


@given(dimension=dimension, num_samples=num_samples, radii=radii())
def test_right_dimensions(dimension, num_samples, radii):
    minor_radius, major_radius = radii
    data = annulus(
        minor_radius=minor_radius,
        major_radius=major_radius,
        dim=dimension,
        num_samples=num_samples,
    )

    assert data.shape == (num_samples, dimension + 1)


@given(dimension=dimension, num_samples=num_samples, radii=radii())
def test_in_annulus(dimension, num_samples, radii):
    minor_radius, major_radius = radii
    data = annulus(
        minor_radius=minor_radius,
        major_radius=major_radius,
        dim=dimension,
        num_samples=num_samples,
    )

    assert np.all(np.linalg.norm(data, axis=1) <= major_radius)
    assert np.all(np.linalg.norm(data, axis=1) >= minor_radius)


@given(dimension=dimension, radii=radii())
def test_mean(dimension, radii):
    minor_radius, major_radius = radii
    data = annulus(
        minor_radius=minor_radius,
        major_radius=major_radius,
        dim=dimension,
        num_samples=100_000,
    )

    assert np.linalg.norm(np.mean(data, axis=0)) < 0.1
