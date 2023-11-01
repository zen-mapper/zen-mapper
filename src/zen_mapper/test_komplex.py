from collections import defaultdict
from functools import reduce
from itertools import chain, combinations, permutations
from math import comb

import pytest
from hypothesis import given
from hypothesis.strategies import frozensets, integers, sets

from zen_mapper.komplex import Komplex, Simplex, _get_candidates, compute_nerve


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


@given(sets(integers(), min_size=1, max_size=15))
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

    for dim in range(0, s.dim + 1):
        assert counter[dim] == comb(s.dim + 1, dim + 1)


def test_candidates():
    prev = {(1, 2), (2, 3), (1, 3), (2, 4)}
    prev = set(map(Simplex, prev))
    test_candidates = set(_get_candidates(prev, 2))
    assert test_candidates == {Simplex([1, 3, 2])}


@given(sets(integers(), min_size=1, max_size=15))
def test_komplex_dimension(v):
    s = Simplex(v)
    k = Komplex(s.faces)
    assert k.dim == s.dim


elements = frozensets(integers(min_value=0, max_value=20), min_size=1, max_size=10)


@given(sets(elements, max_size=5))
def test_nerve(elements: set[frozenset[int]]):
    element_list = list(elements)
    nerve = compute_nerve(element_list, dim=None)
    vertices = range(len(elements))
    candidates = chain.from_iterable(
        combinations(vertices, i) for i in range(1, len(elements) + 1)
    )
    for candidate in candidates:
        intersection = reduce(
            lambda prev, new: prev.intersection(new),
            (element_list[i] for i in candidate),
        )
        if len(intersection):
            assert Simplex(candidate) in nerve
        else:
            assert Simplex(candidate) not in nerve
