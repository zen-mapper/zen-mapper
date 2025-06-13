import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from kaiju_mapper.datasets import torus

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


@given(num_samples=num_samples, radii=radii())
def test_right_dimensions(num_samples, radii):
    minor_radius, major_radius = radii
    data = torus(
        minor_radius=minor_radius,
        major_radius=major_radius,
        num_samples=num_samples,
    )

    assert data.shape == (num_samples, 3)


@given(num_samples=num_samples, radii=radii())
def test_on_torus(num_samples, radii):
    minor_radius, major_radius = radii
    data = torus(
        minor_radius=minor_radius,
        major_radius=major_radius,
        num_samples=num_samples,
    )

    x = (np.linalg.norm(data[:, :-1], axis=1) - major_radius) / minor_radius
    y = data[:, -1] / minor_radius
    theta = np.atan2(y, x)
    phi = np.atan2(data[:, 1], data[:, 0])

    assert np.all(~np.isnan(phi))
    assert np.all(~np.isnan(theta))
