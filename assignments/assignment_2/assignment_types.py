from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt


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

type FitnessType = Literal["baseline"]
type CrossoverType = Literal["none", "one_point", "n_point", "uniform"]

type FloatArray = npt.NDArray[np.float64]
type Position = FloatArray
type Coordinates = tuple[float, float, float]
type Weights = list[FloatArray]
type WeightShapes = list[tuple[int, ...]]
type Genotype = list[float]

type FitnessFunction = Callable[[Position, Position, Position], float]
type FitnessBuilder = Callable[[], FitnessFunction]

# Two flat parents produce two independent flat children.
type CrossoverFunction = Callable[[Genotype, Genotype], tuple[Genotype, Genotype]]
type CrossoverBuilder = Callable[[int, float], CrossoverFunction]
