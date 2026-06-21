from itertools import chain

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from kaiju_mapper.gmapper import _make_interval, _split, g_mapper_cover


# The sklearn GaussianMixture fitting procedure generates a lot of warnings on
# degenerate input, this is expected and we don't really want to hear about it
# in our test suite.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    # Keep means well-separated to help GaussianMixture converge
    left_mean=st.floats(
        min_value=0, max_value=20, allow_nan=False, allow_infinity=False
    ),
    right_mean=st.floats(
        min_value=40, max_value=60, allow_nan=False, allow_infinity=False
    ),
    # Generate asymmetric variances
    scale_ratio=st.floats(min_value=2.0, max_value=5.0),
    base_scale=st.floats(min_value=1.0, max_value=3.0),
    which_larger=st.booleans(),
    cluster_size=st.integers(min_value=200, max_value=500),
    # Use a high overlap to stress test interval bounds
    g_overlap=st.floats(
        min_value=0.75,
        max_value=0.98,
        allow_nan=False,
        exclude_min=True,
        exclude_max=True,
    ),
    random_seed=st.integers(min_value=0, max_value=2**15),
)
def test_split_clamping(
    left_mean,
    right_mean,
    scale_ratio,
    base_scale,
    which_larger,
    cluster_size,
    g_overlap,
    random_seed,
):
    """Test that _split handles asymmetric variances with high overlap.

    This targets the case where clamping via min/max is required:
    - Asymmetric cluster variances (one significantly larger)
    - High g_overlap values (0.7–0.95)

    Without clamping: such cases are often rejected due to safety checks.
    With clamping (min/max): bounds are corrected, allowing valid splits.
    """
    # asymmetric scales
    if which_larger:
        left_scale = base_scale * scale_ratio
        right_scale = base_scale
    else:
        left_scale = base_scale
        right_scale = base_scale * scale_ratio

    # two Gaussian clusters
    np.random.seed(random_seed)
    left_cluster = np.random.normal(loc=left_mean, scale=left_scale, size=cluster_size)
    right_cluster = np.random.normal(
        loc=right_mean, scale=right_scale, size=cluster_size
    )
    data = np.concatenate([left_cluster, right_cluster])

    # an interval spanning the data range
    interval = _make_interval(data, lower_bound=data.min(), upper_bound=data.max())

    # split the interval
    result = _split(interval, data, g_overlap=g_overlap, random_state=random_seed)

    # the split should succeed
    assert result is not None, (
        f"Split with asymmetric variances (scale_ratio={scale_ratio:.1f}) "
        f"and high overlap (g_overlap={g_overlap:.2f}) should succeed with "
        f"proper bounds."
    )


cover_scheme = st.fixed_dictionaries(
    {
        "iterations": st.integers(min_value=1),
        "max_intervals": st.integers(min_value=1, max_value=1_000),
        "g_overlap": st.floats(
            min_value=0,
            max_value=1,
            allow_nan=False,
            exclude_min=True,
            exclude_max=True,
        ),
        "ad_threshold": st.floats(
            min_value=0,
            max_value=250,
            allow_nan=False,
            exclude_min=True,
        ),
    }
)


# The sklearn GaussianMixture fitting procedure generates a lot of warnings on
# degenerate input, this is expected and we don't really want to hear about it
# in our test suite.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    cover_scheme,
    arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=10_000),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    ),
)
def test_max_intervals(cover_scheme: dict, data: np.ndarray):
    cover = g_mapper_cover(data=data, **cover_scheme)
    assert len(cover) <= cover_scheme["max_intervals"]


# The sklearn GaussianMixture fitting procedure generates a lot of warnings on
# degenerate input, this is expected and we don't really want to hear about it
# in our test suite.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    cover_scheme,
    arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=10_000),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    ),
)
def test_coverage(cover_scheme: dict, data: np.ndarray):
    cover = g_mapper_cover(data=data, **cover_scheme)
    covered_points = set(chain(*cover))
    assert covered_points == set(np.arange(len(data)))
