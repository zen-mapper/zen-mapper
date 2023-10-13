from collections import defaultdict
from itertools import permutations
from math import comb

import pytest
from hypothesis import given
from hypothesis.strategies import integers, sets

from . import Simplex


def test_empty():
    with pytest.raises(AssertionError):
        Simplex(())


def test_repeats():
    with pytest.raises(AssertionError):
        Simplex([1, 2, 3, 2])


@given(sets(integers(), min_size=1, max_size=5))
def test_dimension(v):
    s = Simplex(v)
    assert s.dim == len(v) - 1


@given(sets(integers(), min_size=1, max_size=5))
def test_iteration(v):
    s = Simplex(v)
    assert set(s) == v


@given(sets(integers(), min_size=1, max_size=5))
def test_equality(v):
    s = Simplex(v)
    for t in permutations(v):
        assert s == Simplex(t)


@given(sets(integers(), min_size=1, max_size=10))
def test_faces(v):
    s = Simplex(v)
    faces = set(s.faces)

    # No repeated faces
    assert len(faces) == len(list(s.faces))

    # Each face needs to be a subsetof the original vertices
    for face in faces:
        assert set(face) <= v

    # For a dimension d simplex the number of dimension k faces is d+1 choose
    # k+1
    counter = defaultdict(int)
    for face in faces:
        counter[face.dim] += 1

    for dim, count in counter.items():
        assert count == comb(s.dim + 1, dim + 1)
