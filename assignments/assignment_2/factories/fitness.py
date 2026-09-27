"""Build fitness functions by name. All scores are minimized."""

from collections.abc import Callable
from typing import ClassVar

import numpy as np
import numpy.typing as npt


type Position = npt.NDArray[np.float64]
type FitnessFunction = Callable[[Position, Position, Position], float]


class FitnessFactory:
    """Return a scoring function: FitnessFactory(str)(start, end, target)."""

    @staticmethod
    def baseline() -> FitnessFunction:
        """Build the original final-XY-distance fitness function."""
        def evaluate(
            initial_position: Position,
            final_position: Position,
            target_position: Position,
        ) -> float:
            # Keep initial_position in the common interface for future variants.
            return float(np.linalg.norm(final_position[:2] - target_position[:2]))

        return evaluate

    BUILDERS: ClassVar[dict[str, Callable[[], FitnessFunction]]] = {
        "baseline": baseline,
    }

    @staticmethod
    def __new__(cls: object, name: str) -> FitnessFunction:
        
        try:
            builder = cls.BUILDERS[name]
        except KeyError:
            available = ", ".join(cls.BUILDERS)
            raise ValueError(
                f"Unknown fitness {name!r}. Available: {available}"
            ) from None
        return builder()
