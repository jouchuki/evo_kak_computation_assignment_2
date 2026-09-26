"""Example: one-min EA using the refactored prototype of ariel.ec module."""

from typing import cast

from rich.console import Console
from rich.traceback import install

from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    Individual,
    FloatMutator,
    # IntegersGenerator,
    Population,
    config,
    set_seed,
)

import pathlib
import random
import numpy as np


from utils import make_random_weights, flatten_weights, unflatten_weights
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
            *,
            seed: int,
            simulation: RobotSimulation | None = None,
            ) -> None:

        # we need the output folder always set
        assert config_output_folder is not None and isinstance(config_output_folder, pathlib.Path) 

        self.seed = seed
        set_seed(seed)
        random.seed(seed)  # Population.shuffle/sample use python random
        self.rng = np.random.default_rng(seed)

        self.config = config

        # EA configuration
        # TODO: make it also part of settings directly, no need to take them as args
        config.is_maximisation = is_maximisation
        config.target_population_size = target_population_size
        config.num_steps = num_steps
        config.output_folder = config_output_folder

        # sim world
        self.simulation = simulation if simulation is not None else RobotSimulation()
        self.simulation.output_folder = config_output_folder
        self.simulation.rng = self.rng

        # TODO: move all the model and world building logic here basically so that i can have access
        # to input/hidden/output easily from the class itself
        # TODO: migrate to torch nn sequential htis is unbearable

        self.INPUT, self.OUTPUT= self.simulation.INPUT_SIZE, self.simulation.OUTPUT_SIZE
        
        self.weight_shapes = [
            (self.INPUT, self.simulation.HIDDEN_SIZE),
            (self.simulation.HIDDEN_SIZE, self.OUTPUT)
        ]

        self.population = Population([self._make_individual() 
                                      for _ in range(config.target_population_size)])

        self.ops: list[EAOperation[...]] = [
                    EAOperation(self.parent_selection),
                    EAOperation(self.crossover),
                    EAOperation(self.mutate),
                    EAOperation(self.evaluate),
                    EAOperation(self.survivor_selection),
                ]

    def _make_individual(self) -> Individual:
        ind = Individual()
        ind.genotype = flatten_weights(make_random_weights(
            self.INPUT, self.OUTPUT, self.rng,
            hidden_size=self.simulation.HIDDEN_SIZE,
        ))
        return ind

    def evaluate(self, population: Population) -> Population:

        for ind in population.unevaluated:
            weights = unflatten_weights(
                cast(list[float], ind.genotype),
                self.weight_shapes,
            )
            ind.fitness = self.simulation.run_experiment(ind_weights=weights, mode="simple")

        return population

    def parent_selection(self, population: Population) -> Population:

        population = population.shuffle()


        # clear selected tags
        for ind in population:
            ind.tags = {"selected": False}


        for idx in range(0, len(population) - 1, 2):
            ind_a = population[idx]
            ind_b = population[idx + 1]
            if ind_a.fitness_ is not None and ind_b.fitness_ is not None:
                if ind_a.fitness_ <= ind_b.fitness_:
                    ind_a.tags = {"selected": True}
                    ind_b.tags = {"selected": False}
                else:
                    ind_a.tags = {"selected": False}
                    ind_b.tags = {"selected": True}
        return population

    def crossover(self, population: Population) -> Population:
        parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))
        for idx in range(0, len(parents) - 1, 2):
            p_a = parents[idx]
            p_b = parents[idx + 1]
            g_a, g_b = Crossover.one_point(
                cast("list[float]", p_a.genotype),
                cast("list[float]", p_b.genotype),
            )

            # one_point returns float genotypes; keep this encoding integer.
            child_a = Individual()
            child_a.genotype = [float(gene) for gene in g_a]
            child_a.tags = {"mutate": True}

            child_b = Individual()
            child_b.genotype = [float(gene) for gene in g_b]
            child_b.tags = {"mutate": True}

            population.extend([child_a, child_b])
        return population

    def mutate(self, population: Population) -> Population:
        for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
            ind.genotype = FloatMutator.gaussian(
                individual=cast("list[float]", ind.genotype),
                std=0.5,
                mutation_probability=0.2,
            )
            ind.requires_eval = True
            ind.tags = {"mutate": False}
        return population

    def survivor_selection(self, population: Population) -> Population:
        shuffled = population.alive.shuffle()
        alive_count = len(shuffled)
        for idx in range(0, len(shuffled) - 1, 2):
            if alive_count <= config.target_population_size:
                break
            ind_a = shuffled[idx]
            ind_b = shuffled[idx + 1]
            if (ind_a.fitness_ or 0.0) <= (ind_b.fitness_ or 0.0):
                ind_b.alive = False
            else:
                ind_a.alive = False
            alive_count -= 1
        return population

    def run(self) -> Population:
        # survivor_selection culls down to this; keep it in line with the seed size.
        self.population = self.evaluate(self.population)

        self.ea = EA(
            self.population,
            self.ops,
            num_steps=self.config.num_steps,
        )
        self.ea.run()

        # record the best guy
        best = self.ea.get_solution("best", only_alive=False)
        weights = unflatten_weights(
            cast(list[float], best.genotype),
            self.weight_shapes,
        )
        self.simulation.run_experiment(ind_weights=weights, mode="video")
    
        console.log("--- Results ---")
        console.log(f"best = {self.ea.get_solution('best', only_alive=False)}")
        console.log(f"median = {self.ea.get_solution('median', only_alive=False)}")
        console.log(f"worst = {self.ea.get_solution('worst', only_alive=False)}")
    
        # Population API examples
        self.ea.fetch_population(only_alive=False, requires_eval=False)
        db_pop = self.ea.population
        top_10 = db_pop.best(sort="min", attribute="fitness_", n=10)
        console.log(f"top-10 from DB = {top_10}")
    
        sampled_best = db_pop.sample(30).best(sort="min", attribute="fitness_", n=5)
        console.log(f"sample(30).best(n=5) = {sampled_best}")
        return sampled_best
