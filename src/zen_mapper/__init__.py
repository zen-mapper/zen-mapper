from collections.abc import Iterable
from itertools import combinations
from typing import Self


class Simplex:
    def __init__(self: Self, vertices: Iterable[int]) -> None:
        _simplex = sorted(vertices)
        assert len(_simplex) == len(
            set(_simplex)
        ), "A simplex must not have repeated elements"
        assert len(_simplex) != 0, "A simplex must have at least one vertex"

        self._simplex: tuple[int, ...] = tuple(_simplex)

    @property
    def dim(self: Self) -> int:
        return len(self._simplex) - 1

    @property
    def faces(self: Self):
        for i in range(1, len(self._simplex) + 1):
            yield from map(Simplex, combinations(self._simplex, i))

    def __eq__(self: Self, other: Self) -> bool:
        return self._simplex == other._simplex

    def __hash__(self: Self) -> int:
        return hash(self._simplex)

    def __iter__(self: Self):
        yield from self._simplex
