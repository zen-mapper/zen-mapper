from itertools import chain

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from kaiju_mapper.gmapper import GMapperCoverScheme


@st.composite
def cover_scheme(draw):
    return GMapperCoverScheme(
        iterations=draw(st.integers(min_value=1)),
        max_intervals=draw(st.integers(min_value=1, max_value=1_000)),
        g_overlap=draw(
            st.floats(
                min_value=0,
                max_value=1,
                allow_nan=False,
                exclude_min=True,
                exclude_max=True,
            )
        ),
        ad_threshold=draw(
            st.floats(
                min_value=0,
                max_value=250,
                allow_nan=False,
                exclude_min=True,
            )
        ),
    )


# The sklearn GaussianMixture fitting procedure generates a lot of warnings on
# degenerate input, this is expected and we don't really want to hear about it
# in our test suite.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    cover_scheme(),
    arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=10_000),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    ),
)
def test_max_intervals(cover_scheme: GMapperCoverScheme, data: np.ndarray):
    cover = cover_scheme(data)
    assert len(cover) <= cover_scheme.max_intervals


# The sklearn GaussianMixture fitting procedure generates a lot of warnings on
# degenerate input, this is expected and we don't really want to hear about it
# in our test suite.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    cover_scheme(),
    arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=10_000),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    ),
)
def test_coverage(cover_scheme: GMapperCoverScheme, data: np.ndarray):
    cover = cover_scheme(data)
    covered_points = set(chain(*cover))
    assert covered_points == set(np.arange(len(data)))
