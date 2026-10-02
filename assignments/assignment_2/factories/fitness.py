"""Build fitness functions by name. All scores are minimized."""

from typing import ClassVar

import numpy as np

from assignment_types import FitnessBuilder, FitnessFunction, Position


class FitnessFactory:
    """Build distance-only or reach-and-stay fitness."""

    @staticmethod
    def baseline(speed_penalty: float = 0.0) -> FitnessFunction:
        """Build the original final-XY-distance fitness function."""
        def evaluate(
            initial_position: Position,
            final_position: Position,
            target_position: Position,
            first_arrival_seconds: float | None = None,
            simulation_duration: float = 1.0,
            time_inside_target: float = 0.0,
        ) -> float:
            # Keep initial_position in the common interface for future variants.
            return float(np.linalg.norm(final_position[:2] - target_position[:2]))

        return evaluate

    @staticmethod
    def improved(speed_penalty: float = 0.0) -> FitnessFunction:
        def evaluate(
            initial_position: Position,
            final_position: Position,
            target_position: Position,
            first_arrival_seconds: float | None,
            simulation_duration: float,
            time_inside_target: float,
        ) -> float:
            if first_arrival_seconds is None:
                initial_distance = float(np.linalg.norm(initial_position[:2] - target_position[:2]))
                final_distance = float(np.linalg.norm(final_position[:2] - target_position[:2]))
                return 3.0 + final_distance / max(initial_distance, 1e-6)
            duration = max(simulation_duration, 1e-6)
            inside_fraction = float(np.clip(time_inside_target / duration, 0.0, 1.0))
            arrival_fraction = float(np.clip(first_arrival_seconds / duration, 0.0, 1.0))
            return 1.0 - inside_fraction + speed_penalty * arrival_fraction

        return evaluate

    BUILDERS: ClassVar[dict[str, FitnessBuilder]] = {
        "baseline": baseline,
        "improved": improved,
    }

    @staticmethod
    def __new__(
        cls: type["FitnessFactory"], name: str, *,
        speed_penalty: float = 0.0,
    ) -> FitnessFunction:
        
        try:
            builder = cls.BUILDERS[name]
        except KeyError:
            available = ", ".join(cls.BUILDERS)
            raise ValueError(
                f"Unknown fitness {name!r}. Available: {available}"
            ) from None
        if not np.isfinite(speed_penalty) or not 0.0 <= speed_penalty <= 1.0:
            raise ValueError("speed_penalty must be between 0 and 1 so success always ranks first")
        return builder(speed_penalty)
