"""Headless evaluation; each spawned worker owns its MuJoCo model and callback."""

import contextlib
import io
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from assignment_types import EvaluationResult, Genotype
from settings import ExperimentConfig
from simulation import RobotSimulation
from utils import unflatten_weights


_worker_simulation: RobotSimulation | None = None


def initialize_worker(config: dict) -> None:
    global _worker_simulation
    with contextlib.redirect_stdout(io.StringIO()):
        _worker_simulation = RobotSimulation(ExperimentConfig(**config))


def evaluate_genome(simulation: RobotSimulation, genes: Genotype) -> EvaluationResult:
    weights = unflatten_weights(genes, [
        (simulation.INPUT_SIZE, simulation.HIDDEN_SIZE),
        (simulation.HIDDEN_SIZE, simulation.OUTPUT_SIZE),
    ])
    score = simulation.run_experiment(weights, mode="simple", quiet=True)
    metrics = dict(simulation.last_metrics)
    if not np.isfinite(score) or any(
        value is not None and not np.isfinite(value) for value in metrics.values()
    ):
        raise ValueError("Non-finite rollout result; stopping")
    return score, metrics


def evaluate_worker(genes: Genotype) -> EvaluationResult:
    if _worker_simulation is None:
        raise RuntimeError("Evaluation worker was not initialized")
    return evaluate_genome(_worker_simulation, genes)


class Evaluator:
    def __init__(self, simulation: RobotSimulation, workers: int = 1) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        # Generated terrains need a shared saved terrain before separate processes
        # can be guaranteed to evaluate the same world.
        if workers > 1 and simulation.settings.WORLD != "SimpleFlatWorld":
            raise ValueError("Parallel evaluation currently requires SimpleFlatWorld")
        self.simulation = simulation
        self.pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=(simulation.settings.model_dump(),),
        ) if workers > 1 else None

    def __call__(self, genomes: list[Genotype]) -> list[EvaluationResult]:
        if self.pool is None:
            return [evaluate_genome(self.simulation, genes) for genes in genomes]
        # Ordered results keep selection and RNG consumption independent of scheduling.
        return list(self.pool.map(evaluate_worker, genomes, chunksize=1))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
