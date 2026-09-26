"""Example: one-max EA using the refactored prototype of ariel.ec module."""

from typing import cast

from rich.console import Console
from rich.traceback import install

from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    Individual,
    IntegerMutator,
    IntegersGenerator,
    Population,
    config,
)

import pathlib


from utils import make_random_weights
from simulation import RobotSimulation

install()
console = Console()


class RobotEA:
    """
    class to orchestrate the population creation and orchestration
    """
    def __init__(
            self,
            config_output_folder: pathlib.Path,
            is_maximisation: bool = False,
            target_population_size: int = 10,
            num_steps: int = 200,
            ) -> None:

        # we need the output folder always set
        assert config_output_folder is not None and isinstance(config_output_folder, pathlib.Path) 

        self.config = config

        # EA configuration
        # TODO: make it also part of settings directly, no need to take them as args
        config.is_maximisation = is_maximisation
        config.target_population_size = target_population_size
        config.num_steps = num_steps
        config.output_folder = config_output_folder

        # sim world
        self.simulation = RobotSimulation()

        # TODO: move all the model and world building logic here basically so that i can have access
        # to input/hidden/output easily from the class itself

        self.INPUT, self.OUTPUT= self.simulation.INPUT_SIZE, self.simulation.OUTPUT_SIZE

        self.population = Population([self._make_individual() 
                                      for _ in range(config.target_population_size)])

        self.ea = EA()

    def _make_individual(self) -> Individual:
        # TODO: need to be able to flatten and unflatten the weights.
        ind = Individual()
        ind.genotype = make_random_weights(self.INPUT, self.OUTPUT)
        return ind

    def evaluate(self) -> None:
        self.simulation.run_experiment()
        pass

    def parent_selection(self) -> None:
        pass

    def crossover(self) -> None:
        pass

    def mutate(self) -> None:
        pass

    def survivor_selection(self) -> None:
        pass




# ── Individual factory ────────────────────────────────────────────────────────

def make_individual() -> Individual:
    ind = Individual()
    ind.genotype = IntegersGenerator.integers(low=0, high=1, size=10)
    return ind


# ── EA steps ─────────────────────────────────────────────────────────────────

def evaluate(population: Population) -> Population:
    for ind in population.unevaluated:
        ind.fitness = float(sum(1 for gene in ind.genotype if gene == 1))
    return population


def parent_selection(population: Population) -> Population:

    shuffled = population.shuffle()
    for idx in range(0, len(shuffled) - 1, 2):
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if ind_a.fitness_ is not None and ind_b.fitness_ is not None:
            if ind_a.fitness_ >= ind_b.fitness_:
                ind_a.tags = {"selected": True}
                ind_b.tags = {"selected": False}
            else:
                ind_a.tags = {"selected": False}
                ind_b.tags = {"selected": True}

    return shuffled


def crossover(population: Population) -> Population:
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))
    for idx in range(0, len(parents) - 1, 2):
        p_a = parents[idx]
        p_b = parents[idx + 1]
        g_a, g_b = Crossover.one_point(
            cast("list[int]", p_a.genotype),
            cast("list[int]", p_b.genotype),
        )

        # one_point returns float genotypes; keep this encoding integer.
        child_a = Individual()
        child_a.genotype = [int(gene) for gene in g_a]
        child_a.tags = {"mutate": True}

        child_b = Individual()
        child_b.genotype = [int(gene) for gene in g_b]
        child_b.tags = {"mutate": True}

        population.extend([child_a, child_b])
    return population


def mutate(population: Population) -> Population:
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
        ind.genotype = IntegerMutator.integer_creep(
            individual=cast("list[int]", ind.genotype),
            span=1,
            mutation_probability=0.2,
        )
        ind.requires_eval = True
    return population


def survivor_selection(population: Population) -> Population:
    shuffled = population.alive.shuffle()
    alive_count = len(shuffled)
    for idx in range(0, len(shuffled) - 1, 2):
        if alive_count <= config.target_population_size:
            break
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if (ind_a.fitness_ or 0.0) >= (ind_b.fitness_ or 0.0):
            ind_b.alive = False
        else:
            ind_a.alive = False
        alive_count -= 1
    return population


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # survivor_selection culls down to this; keep it in line with the seed size.
    config.target_population_size = 20

    initial = Population([make_individual() for _ in range(20)])
    initial = evaluate(initial)

    ops: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
    ]

    ea = EA(initial, ops, num_steps=50)
    ea.run()

    console.log("--- Results ---")
    console.log(f"best = {ea.get_solution('best', only_alive=False)}")
    console.log(f"median = {ea.get_solution('median', only_alive=False)}")
    console.log(f"worst = {ea.get_solution('worst', only_alive=False)}")

    # Population API examples
    db_pop = ea.fetch(only_alive=False)
    top_10 = db_pop.best(sort="max", attribute="fitness_", n=10)
    console.log(f"top-10 from DB = {top_10}")

    sampled_best = db_pop.sample(30).best(sort="max", attribute="fitness_", n=5)
    console.log(f"sample(30).best(n=5) = {sampled_best}")


if __name__ == "__main__":
    main()
