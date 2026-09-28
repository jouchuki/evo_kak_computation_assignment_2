"""Build fitness functions by name. All scores are minimized."""

from typing import ClassVar

import numpy as np


from assignment_types import FitnessBuilder, FitnessFunction, Position


class FitnessFactory:
    """Return a scoring function: FitnessFactory(str)(start, end, target, on_platform)."""

    @staticmethod
    def baseline(fall_penalty: float = 10.0) -> FitnessFunction:
        """Build the original final-XY-distance fitness function."""
        def evaluate(
            initial_position: Position,
            final_position: Position,
            target_position: Position,
            on_platform: bool = True,
        ) -> float:
            # Keep initial_position in the common interface for future variants.
            return float(np.linalg.norm(final_position[:2] - target_position[:2]))

        return evaluate

    @staticmethod
    def improved(fall_penalty: float = 10.0) -> FitnessFunction:
        def evaluate(
            initial_position: Position,
            final_position: Position,
            target_position: Position,
            on_platform: bool,
        ) -> float:
            distance = float(np.linalg.norm(final_position[:2] - target_position[:2]))
            return distance + (0.0 if on_platform else fall_penalty)

        return evaluate

    BUILDERS: ClassVar[dict[str, FitnessBuilder]] = {
        "baseline": baseline,
        "improved": improved,
    }

    @staticmethod
    def __new__(
        cls: type["FitnessFactory"], name: str, *, fall_penalty: float = 10.0,
    ) -> FitnessFunction:
        
        try:
            builder = cls.BUILDERS[name]
        except KeyError:
            available = ", ".join(cls.BUILDERS)
            raise ValueError(
                f"Unknown fitness {name!r}. Available: {available}"
            ) from None
        if not np.isfinite(fall_penalty) or fall_penalty < 0:
            raise ValueError("fall_penalty must be finite and nonnegative")
        return builder(fall_penalty)
