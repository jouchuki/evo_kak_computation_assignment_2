"""Build mutation functions for flat neural-network weight genotypes."""

from math import isfinite
from typing import ClassVar

from ariel.ec import FloatMutator

from assignment_types import Genotype, MutateBuilder, MutateFunction


class MutateFactory:
    """Select mutation once, then call the returned function with a genotype."""

    @staticmethod
    def gaussian(probability: float, std: float, low: float, high: float) -> MutateFunction:
        """Add zero-mean Gaussian noise to each selected weight."""
        if not isfinite(std) or std < 0.0:
            raise ValueError("std must be finite and nonnegative")

        def mutate(genes: Genotype) -> Genotype:
            return list(FloatMutator.gaussian(
                individual=genes,
                std=std,
                mutation_probability=probability,
            ))

        return mutate

    @staticmethod
    def uniform_reset(probability: float, std: float, low: float, high: float) -> MutateFunction:
        """Replace selected weights with new values sampled from [low, high)."""
        if not isfinite(low) or not isfinite(high) or low >= high:
            raise ValueError("Uniform reset requires finite bounds with low < high")

        def mutate(genes: Genotype) -> Genotype:
            return list(FloatMutator.uniform_reset(
                individual=genes,
                low=low,
                high=high,
                mutation_probability=probability,
            ))

        return mutate

    @staticmethod
    def none(probability: float, std: float, low: float, high: float) -> MutateFunction:
        """Return an independent copy without changing any weights."""
        def mutate(genes: Genotype) -> Genotype:
            return genes.copy()

        return mutate

    BUILDERS: ClassVar[dict[str, MutateBuilder]] = {
        "gaussian": gaussian,
        "uniform_reset": uniform_reset,
        "none": none,
    }

    @staticmethod
    def __new__(
        cls: type["MutateFactory"],
        name: str,
        *,
        probability: float = 0.2,
        std: float = 0.5,
        low: float = -1.0,
        high: float = 1.0,
    ) -> MutateFunction:
        try:
            builder = cls.BUILDERS[name]
        except KeyError:
            available = ", ".join(cls.BUILDERS)
            raise ValueError(
                f"Unknown mutation {name!r}. Available: {available}"
            ) from None
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        return builder(probability, std, low, high)
