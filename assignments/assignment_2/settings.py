from pydantic_settings import BaseSettings
from pydantic import Field
# import numpy as np
from pathlib import Path
import yaml
from assignment_types import AllowedRobotTypes, AllowedWorldTypes, CrossoverType, FitnessType, MutateType, ViewerTypes

class ExperimentConfig(BaseSettings):

    SEEDS: list[int] = [42, 321, 1337, 313, 58]

    ROBOT: AllowedRobotTypes = "turtle"
    WORLD:  AllowedWorldTypes = "OlympicArena"

    # --- DATA SETUP --- #
    CWD: Path = Path.cwd()

    SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts
    # TARGET_POS moved to simulation.py
    SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
    MODE: ViewerTypes = "launcher"  # see run_experiment() for the options

    HIDDEN_SIZE: int = 6
    FITNESS_TYPE: FitnessType = "baseline"

    # population and run settings

    TARGET_POPULATION_SIZE: int = 10
    NUM_STEPS: int = 200
    RUN_NAME: str = ""


    # crossover settings
    CROSSOVER_TYPE: CrossoverType = "one_point"
    CROSSOVER_N_POINTS: int = Field(default=2, ge=1)
    CROSSOVER_SWAP_PROB: float = Field(default=0.5, ge=0.0, le=1.0)

    # mutation settings
    MUTATION_TYPE: MutateType = "gaussian"
    MUTATION_PROB: float = Field(default=0.2, ge=0.0, le=1.0)
    MUTATION_STD: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    MUTATION_LOW: float = Field(default=-1.0, allow_inf_nan=False)
    MUTATION_HIGH: float = Field(default=1.0, allow_inf_nan=False)

    @property
    def SCRIPT_NAME(self) -> str:
        return f"{self.ROBOT}_{self.WORLD}"

    @property
    def DATA(self):
        # use base if no run - overwrites the root run
        base = self.CWD / "__data__" / self.SCRIPT_NAME
        # if run name supplied - mkdir a new folder in base and put all run files there
        return base / self.RUN_NAME if self.RUN_NAME else base

# path to config (change the final filename to our config from __file__)
config_path = Path(__file__).with_name("config.yaml")

values = (
    (yaml.safe_load(config_path.read_text()) or {})
    if config_path.exists()
    else {}
)

settings = ExperimentConfig(**values)