from typing import ClassVar, cast

from ariel.ec import Crossover

from assignment_types import CrossoverBuilder, CrossoverFunction, Genotype


class CrossoverFactory:
    """Select and configure crossover once, then call it with two parents."""

    @staticmethod
    def one_point(n_points: int, swap_probability: float) -> CrossoverFunction:
        def crossover(parent_a: Genotype, parent_b: Genotype) -> tuple[Genotype, Genotype]:
            return cast(tuple[Genotype, Genotype], Crossover.one_point(parent_a, parent_b))
        
        return crossover

    @staticmethod
    def n_point(n_points: int, swap_probability: float) -> CrossoverFunction:
        if n_points < 1:
            raise ValueError("n_points must be at least 1")

        def crossover(parent_a: Genotype, parent_b: Genotype) -> tuple[Genotype, Genotype]:
            return cast(
                tuple[Genotype, Genotype],
                Crossover.n_point(parent_a, parent_b, n=n_points),
            )
        return crossover

    @staticmethod
    def uniform(n_points: int, swap_probability: float) -> CrossoverFunction:
        """Bind the probability of swapping each corresponding pair of genes."""
        if not 0.0 <= swap_probability <= 1.0:
            raise ValueError("swap_probability must be between 0 and 1")

        def crossover(parent_a: Genotype, parent_b: Genotype) -> tuple[Genotype, Genotype]:
            return cast(
                tuple[Genotype, Genotype],
                Crossover.uniform(parent_a, parent_b, swap_probability=swap_probability),
            )

        return crossover

    @staticmethod
    def none(n_points: int, swap_probability: float) -> CrossoverFunction:
        """Copy parents into independent children without mixing their genes."""
        def crossover(parent_a: Genotype, parent_b: Genotype) -> tuple[Genotype, Genotype]:
            return parent_a.copy(), parent_b.copy()

        return crossover

    # All builders accept the same configuration arguments and return the same interface.
    BUILDERS: ClassVar[dict[str, CrossoverBuilder]] = {
        "none": none,
        "one_point": one_point,
        "n_point": n_point,
        "uniform": uniform,
    }

    @staticmethod
    def __new__(
        cls: type["CrossoverFactory"],
        name: str,
        *,
        n_points: int = 2,
        swap_probability: float = 0.5,
    ) -> CrossoverFunction:
        try:
            builder = cls.BUILDERS[name]
        except KeyError:
            available = ", ".join(cls.BUILDERS)
            raise ValueError(
                f"Unknown crossover {name!r}. Available: {available}"
            ) from None
        return builder(n_points, swap_probability)
