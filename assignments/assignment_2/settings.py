from pydantic_settings import BaseSettings
# import numpy as np
from pathlib import Path
import yaml
from typing import Literal

type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]
type AllowedWorldTypes = Literal["SimpleFlatWorld","RuggedTerrainWorld", "CraterTerrainWorld", "AmphitheatreTerrainWorld", "OlympicArena"]
type AllowedRobotTypes = Literal[
    "baby_a",
    "baby_b",
    "gecko",
    "linkin_modified",
    "snake",
    "turtle",
    "iguana",
    "spider_8",
    "spider_12",
    "spider_16",
    "centipede_3",
    "centipede_4",
    "centipede_5",
    ]



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

    TARGET_POPULATION_SIZE: int = 10
    NUM_STEPS: int = 200
    RUN_NAME: str = ""


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