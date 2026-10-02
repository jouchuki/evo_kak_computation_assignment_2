"""Run experiments.yaml: uv run python assignments/assignment_2/run_experiments.py"""

import contextlib
import os
from pathlib import Path
import time
import traceback

# Each evaluation worker uses one numerical-library thread.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import yaml

from ariel.ec import EA, EAOperation, Population
from evaluation import Evaluator
from robot_ea import RobotEA
from settings import ExperimentConfig
from simulation import RobotSimulation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run_experiment(cfg: ExperimentConfig, method: str, seed: int, workers: int, folder: Path) -> None:
    started = time.perf_counter()
    simulation = RobotSimulation(cfg, output_folder=folder)
    with Evaluator(simulation, workers) as evaluator:
        robot = RobotEA(
            folder, target_population_size=cfg.TARGET_POPULATION_SIZE,
            num_steps=cfg.NUM_STEPS, seed=seed,
            simulation=simulation, evaluation_map=evaluator,
        )
        robot.config.db_handling = "halt"
        robot.config.quiet = False
        robot.config.first_generation_id = 0
        budget = cfg.TARGET_POPULATION_SIZE + robot.offspring_size * cfg.NUM_STEPS
        with (folder / "config.yaml").open("x") as config_file:
            yaml.safe_dump({
                "method": method, "seed": seed, "workers": workers,
                "evaluation_budget": budget, "settings": cfg.model_dump(mode="json"),
            }, config_file, sort_keys=False)
        try:
            if method == "random":
                # Fresh samples from the same initializer, with the same evaluation budget.
                def resample(population: Population) -> Population:
                    for ind in population:
                        ind.alive = False
                    population.extend([robot.make_individual() for _ in range(robot.offspring_size)])
                    return robot.evaluate(population)

                population = robot.evaluate(robot.population)
                robot.ea = EA(population, [EAOperation(resample)], num_steps=cfg.NUM_STEPS)
                robot.ea.run()
            else:
                robot.run(render_video=False, report_examples=False)
            if robot.evaluation_count != budget:
                raise RuntimeError("Actual evaluation count differs from planned budget")
            print(f"Completed {budget} evaluations in {time.perf_counter() - started:.2f} seconds", flush=True)
        finally:
            if hasattr(robot, "ea"):
                robot.ea.engine.dispose()


def main(study_path: Path = HERE / "experiments.yaml") -> None:
    study = yaml.safe_load(study_path.read_text())
    base = yaml.safe_load((study_path.parent / study["base"]).read_text()) or {}
    base.update(study.get("overrides", {}))
    seeds, methods, workers = study["seeds"], study["methods"], study["workers"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Choose distinct nonnegative integer seeds")
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= {"baseline", "improved", "random"}:
        raise ValueError("Choose distinct methods from baseline, improved, random")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    name = study["name"]
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
        raise ValueError("Study name must be a single directory name")
    weight = float(study["alpha"])
    if not np.isfinite(weight) or not 0 < weight <= 1:
        raise ValueError("alpha must be in (0, 1]")
    cfg = ExperimentConfig(**base)
    if cfg.WORLD != "SimpleFlatWorld":
        raise ValueError("This study uses SimpleFlatWorld; random terrain needs a shared snapshot")
    if cfg.TARGET_POPULATION_SIZE < 2 or cfg.NUM_STEPS < 1 or not np.isfinite(cfg.SIM_DURATION) or cfg.SIM_DURATION <= 0:
        raise ValueError("Use population >= 2, generations >= 1 and positive finite duration")
    root = ROOT / study["output"] / name
    for seed in seeds:
        for method in methods:
            folder = root / method / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier run.
            values = cfg.model_dump()
            values.update(
                SEEDS=[seed], MODE="simple", CWD=ROOT, RUN_NAME=name,
                FITNESS_TYPE="improved", SPEED_PENALTY=weight if method == "improved" else 0.0,
            )
            print(f"Starting {method}, seed {seed}; log: {folder / 'training.log'}", flush=True)
            with (folder / "training.log").open("x", buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    try:
                        run_experiment(ExperimentConfig(**values), method, seed, workers, folder)
                    except BaseException:
                        traceback.print_exc()
                        raise
            print(f"Completed {method}, seed {seed}", flush=True)


if __name__ == "__main__":
    main()
