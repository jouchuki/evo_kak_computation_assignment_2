"""EC A2 template code - neuroevolution for targeted locomotion with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo*, not a solution. It spawns a robot, drives it with a neural network
whose weights are RANDOM, runs the simulation, and reports how close the robot
ended up to a target.

There is deliberately NO evolution in here. Building the EA (representation,
initialisation, parent selection, variation, survivor selection) is the assignment.
See "YOUR JOB" at the bottom of this file.

THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve the weights of a neural network controller so that a robot moves from
SPAWN_POS to TARGET_POSITION within the simulation time.

    fitness = distance between the robot's final position and TARGET_POSITION

HOW TO RUN
----------

Change MODE below to switch between an interactive viewer, a headless run,
a rendered video, or a single frame.
"""

from settings import settings
import numpy as np

# Each RobotEA seeds its RNGs before creating its population.

# --- DATA SETUP --- #
SCRIPT_NAME = settings.SCRIPT_NAME
CWD = settings.CWD
DATA = settings.DATA

DATA.mkdir(parents=True, exist_ok=True)




# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #


# in simulation.py

# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
#
# MuJoCo calls the controller every physics step with (model, data); its job
# is to write into data.ctrl.
#
#   INPUTS   : whatever you read from `data` (qpos, qvel, time, ...), plus any
#              task info you already know, e.g. the vector to TARGET_POSITION.
#              INPUT SIZE is your choice, but must stay CONSTANT.
#   OUTPUTS  : exactly `model.nu` values, one per actuated hinge.
#   RANGE    : hinges accept [-pi/2, +pi/2] radians. A tanh output gives
#              [-1, 1] - rescale: actions * (np.pi / 2).
#   WRITING  : DIRECT (data.ctrl[:] = actions) commands the angle straight -
#              fast, but can destabilise the sim on large jumps. DELTA
#              (data.ctrl[:] += actions * alpha, alpha ~ 0.05, then clip) is
#              smoother but accumulates, so clipping is required. Pick one,
#              justify it, use it everywhere.
#   NaN      : blown-up weights silently write NaN into data.ctrl. Assert
#              against it while developing.
#
# ============================================================================ #


# ============================================================================ #
#  3. POSITION AND FITNESS
# ============================================================================ #
#
# The robot is spawned with a free joint, so data.qpos[0:3] IS the core's
# (x, y, z) world position. Read it before and after stepping - no tracker or
# bookkeeping needed. (`data.geom("robot1_core").xpos` works too.)
#
# ============================================================================ #





# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #





def main() -> None:
    """Run independent evolution seeds on one shared terrain."""
    from robot_ea import RobotEA
    from simulation import RobotSimulation

    if not settings.SEEDS:
        raise ValueError("SEEDS must contain at least one seed")
    if len(set(settings.SEEDS)) != len(settings.SEEDS):
        raise ValueError("SEEDS must be unique so run folders cannot collide")

    # Construct terrain once; run_experiment resets physics for each candidate.
    simulation = RobotSimulation(settings)
    best_scores: list[float] = []
    for seed in settings.SEEDS:
        run_dir = settings.DATA / f"seed_{seed}"
        print(f"Starting seed {seed}; output: {run_dir}")
        ea = RobotEA(
            run_dir,
            target_population_size=settings.TARGET_POPULATION_SIZE,
            num_steps=settings.NUM_STEPS,
            seed=seed,
            simulation=simulation,
        )
        ea.run()
        best = ea.ea.get_solution("best", only_alive=False)
        best_scores.append(float(best.fitness))
        ea.ea.engine.dispose()

    print("Best fitness by seed:", dict(zip(settings.SEEDS, best_scores)))
    print(f"Mean best fitness: {np.mean(best_scores):.6f}")
    if len(best_scores) > 1:
        print(f"Sample standard deviation: {np.std(best_scores, ddof=1):.6f}")

if __name__ == "__main__":
    main()


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above runs one robot with random weights. It will score badly, and
# it will score badly in a slightly different way every time you change SEED.
# Your task is to replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing. Repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline. Random search with the same evaluation
#     budget is a simple, but reasonable choice; and it is nearly free to run.
#   Keep body, world, SIM_DURATION and fitness function identical across
#     everything you compare. Change one thing at a time.
#
# ============================================================================ #
