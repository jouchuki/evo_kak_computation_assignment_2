import random
from math import isfinite
from typing import ClassVar

from ariel.ec import Population

from assignment_types import MatingPool, SelectionFunction, SurvivorSelectionBuilder


class SurvivorSelectionFactory:

    @staticmethod
    def _candidates(population: Population) -> MatingPool:
        candidates = list(population.alive)
        for ind in candidates:
            if ind.fitness_ is None or not isfinite(ind.fitness_):
                raise ValueError("Survivor selection requires finite fitness for every living individual")
        return candidates

    @staticmethod
    def round_robin_tournament(
        target_size: int, num_opponents: int, is_maximisation: bool,
    ) -> SelectionFunction:
        def select(population: Population) -> Population:
            candidates = SurvivorSelectionFactory._candidates(population)
            if len(candidates) <= target_size:
                return population
            if num_opponents >= len(candidates):
                raise ValueError("Round-robin q must be smaller than the combined living population")
            scores: list[int] = []
            for idx, ind in enumerate(candidates):
                opponents = random.sample(
                    candidates[:idx] + candidates[idx + 1:], k=num_opponents,
                )
                wins = sum(
                    ind.fitness > other.fitness if is_maximisation
                    else ind.fitness < other.fitness
                    for other in opponents
                )
                scores.append(wins)
            ranked = list(zip(candidates, scores))
            random.shuffle(ranked)
            ranked.sort(key=lambda entry: entry[1], reverse=True)
            for ind, _ in ranked[target_size:]:
                ind.alive = False
            return population

        return select

    @staticmethod
    def mu_plus_lambda(
        target_size: int, num_opponents: int, is_maximisation: bool,
    ) -> SelectionFunction:
        def select(population: Population) -> Population:
            candidates = SurvivorSelectionFactory._candidates(population)
            random.shuffle(candidates)  # Random ordering among equal fitness values.
            candidates.sort(key=lambda ind: ind.fitness, reverse=is_maximisation)
            for ind in candidates[target_size:]:
                ind.alive = False
            return population

        return select

    BUILDERS: ClassVar[dict[str, SurvivorSelectionBuilder]] = {
        "round_robin_tournament": round_robin_tournament,
        "mu_plus_lambda": mu_plus_lambda,
    }

    @staticmethod
    def __new__(
        cls: type["SurvivorSelectionFactory"], name: str, *, target_size: int,
        num_opponents: int = 10, is_maximisation: bool = False,
    ) -> SelectionFunction:
        try:
            builder = cls.BUILDERS[name]
        except KeyError:
            available = ", ".join(cls.BUILDERS)
            raise ValueError(f"Unknown survivor selection {name!r}. Available: {available}") from None
        if target_size < 1 or num_opponents < 1:
            raise ValueError("target_size and num_opponents must be positive")
        return builder(target_size, num_opponents, is_maximisation)
