"""Example: one-min EA using the refactored prototype of ariel.ec module."""

from typing import cast

from assignment_types import Genotype, MatingPool, WeightShapes

from rich.console import Console
from rich.traceback import install

from ariel.ec import (
    EA,
    EAOperation,
    Individual,
    # IntegersGenerator,
    Population,
    config,
    set_seed,
)

import pathlib
import random
import numpy as np


from utils import make_random_weights, make_xavier_weights, flatten_weights, unflatten_weights
from simulation import RobotSimulation
from settings import settings
from factories import (
    CrossoverFactory, MutateFactory, ParentSelectionFactory, SurvivorSelectionFactory,
)

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

        # Selection and variation policies.
        experiment_settings = self.simulation.settings
        self.offspring_size = experiment_settings.OFFSPRING_SIZE
        if self.offspring_size is None:
            self.offspring_size = max(2, 2 * (target_population_size // 4))
        self.mating_pool: MatingPool = []
        self._parent_selection = ParentSelectionFactory(
            experiment_settings.PARENT_SELECTION_TYPE,
            num_parents=self.offspring_size,
            tournament_size=experiment_settings.TOURNAMENT_SIZE,
            with_replacement=experiment_settings.TOURNAMENT_WITH_REPLACEMENT,
            is_maximisation=is_maximisation,
        )

        # SURVIVOR SELECTION
        self._survivor_selection = SurvivorSelectionFactory(
            experiment_settings.SURVIVOR_SELECTION_TYPE,
            target_size=target_population_size,
            num_opponents=experiment_settings.ROUND_ROBIN_OPPONENTS,
            is_maximisation=is_maximisation,
        )

        # CROSSOVER
        self._crossover = CrossoverFactory(
            experiment_settings.CROSSOVER_TYPE,
            n_points=experiment_settings.CROSSOVER_N_POINTS,
            swap_probability=experiment_settings.CROSSOVER_SWAP_PROB,
        )

        # MUTATION
        self._mutate = MutateFactory(
            experiment_settings.MUTATION_TYPE,
            probability=experiment_settings.MUTATION_PROB,
            std=experiment_settings.MUTATION_STD,
            low=experiment_settings.MUTATION_LOW,
            high=experiment_settings.MUTATION_HIGH,
        )

        # TODO: move all the model and world building logic here basically so that i can have access
        # to input/hidden/output easily from the class itself
        # TODO: migrate to torch nn sequential htis is unbearable

        self._make_weights = (
            make_xavier_weights
            if experiment_settings.WEIGHT_INIT == "xavier_uniform"
            else make_random_weights
        )

        self.INPUT, self.OUTPUT= self.simulation.INPUT_SIZE, self.simulation.OUTPUT_SIZE
        
        self.weight_shapes: WeightShapes = [
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
        ind.genotype = flatten_weights(self._make_weights(
            self.INPUT, self.OUTPUT, self.rng,
            hidden_size=self.simulation.HIDDEN_SIZE,
        ))
        return ind

    def evaluate(self, population: Population) -> Population:

        for ind in population.unevaluated:
            weights = unflatten_weights(
                cast(Genotype, ind.genotype),
                self.weight_shapes,
            )
            ind.fitness = self.simulation.run_experiment(ind_weights=weights, mode=settings.MODE)

        return population

    def parent_selection(self, population: Population) -> Population:
        self.mating_pool = self._parent_selection(population)
        return population

    def crossover(self, population: Population) -> Population:
        parents = self.mating_pool
        for idx in range(0, len(parents) - 1, 2):
            p_a = parents[idx]
            p_b = parents[idx + 1]
            g_a, g_b = self._crossover(
                cast(Genotype, p_a.genotype),
                cast(Genotype, p_b.genotype),
            )

            # Each child stores a flat, JSON-compatible list of float weights.
            child_a = Individual()
            child_a.genotype = [float(gene) for gene in g_a]
            child_a.tags = {"mutate": True}

            child_b = Individual()
            child_b.genotype = [float(gene) for gene in g_b]
            child_b.tags = {"mutate": True}

            # we add children here, then remove at survivor selection stage
            population.extend([child_a, child_b])
        self.mating_pool = []
        return population

    def mutate(self, population: Population) -> Population:
        for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
            ind.genotype = self._mutate(cast(Genotype, ind.genotype))
            ind.requires_eval = True
            ind.tags = {"mutate": False}
        return population

    def survivor_selection(self, population: Population) -> Population:
        return self._survivor_selection(population)

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
            cast(Genotype, best.genotype),
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
