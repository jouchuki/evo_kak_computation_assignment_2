import random
from math import isfinite
from typing import ClassVar

from ariel.ec import Population

from assignment_types import MatingPool, ParentSelectionBuilder, ParentSelectionFunction


class ParentSelectionFactory:

    @staticmethod
    def _candidates(population: Population) -> MatingPool:
        candidates = list(population.alive)
        if not candidates:
            raise ValueError("Parent selection requires a living population")
        for ind in candidates:
            if ind.fitness_ is None or not isfinite(ind.fitness_):
                raise ValueError("Parent selection requires finite fitness for every living individual")
        return candidates

    @staticmethod
    def tournament(
        num_parents: int, tournament_size: int, with_replacement: bool,
        is_maximisation: bool,
    ) -> ParentSelectionFunction:
        def select(population: Population) -> MatingPool:
            candidates = ParentSelectionFactory._candidates(population)
            if not with_replacement and tournament_size > len(candidates):
                raise ValueError("Tournament size exceeds the living population without replacement")
            choose_best = max if is_maximisation else min
            parents: MatingPool = []
            for _ in range(num_parents):
                contestants = (
                    random.choices(candidates, k=tournament_size)
                    if with_replacement
                    else random.sample(candidates, k=tournament_size)
                )
                parents.append(choose_best(contestants, key=lambda ind: ind.fitness))
            return parents

        return select

    @staticmethod
    def uniform(
        num_parents: int, tournament_size: int, with_replacement: bool,
        is_maximisation: bool,
    ) -> ParentSelectionFunction:
        def select(population: Population) -> MatingPool:
            candidates = ParentSelectionFactory._candidates(population)
            return random.choices(candidates, k=num_parents)

        return select

    BUILDERS: ClassVar[dict[str, ParentSelectionBuilder]] = {
        "tournament": tournament,
        "uniform": uniform,
    }

    @staticmethod
    def __new__(
        cls: type["ParentSelectionFactory"], name: str, *, num_parents: int,
        tournament_size: int = 2, with_replacement: bool = False,
        is_maximisation: bool = False,
    ) -> ParentSelectionFunction:
        try:
            builder = cls.BUILDERS[name]
        except KeyError:
            available = ", ".join(cls.BUILDERS)
            raise ValueError(f"Unknown parent selection {name!r}. Available: {available}") from None
        if num_parents < 1 or tournament_size < 1:
            raise ValueError("num_parents and tournament_size must be positive")
        return builder(num_parents, tournament_size, with_replacement, is_maximisation)
