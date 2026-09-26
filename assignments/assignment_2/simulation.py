from settings import settings, ExperimentConfig, ViewerTypes
from ariel.simulation.environments import BaseWorld, SimpleFlatWorld, RuggedTerrainWorld, AmphitheatreTerrainWorld, OlympicArena, CraterTerrainWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import (
    baby_a,
    baby_b,
    gecko,
    linkin_modified,
    snake,
    turtle,
    iguana,
    spider_8,
    spider_12,
    spider_16,
    centipede_3,
    centipede_4,
    centipede_5,
)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel import console

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# moduels
from utils import make_random_weights

WORLD_TYPES = {
    "SimpleFlatWorld": SimpleFlatWorld,
    "RuggedTerrainWorld": RuggedTerrainWorld,
    "AmphitheatreTerrainWorld": AmphitheatreTerrainWorld,
    "OlympicArena": OlympicArena,
    "CraterTerrainWorld": CraterTerrainWorld
}

ROBOT_TYPES = {
    "baby_a": baby_a,
    "baby_b": baby_b,
    "gecko": gecko,
    "linkin_modified": linkin_modified,
    "snake": snake,
    "turtle": turtle,
    "iguana": iguana,
    "spider_8": spider_8,
    "spider_12": spider_12,
    "spider_16": spider_16,
    "centipede_3": centipede_3,
    "centipede_4": centipede_4,
    "centipede_5": centipede_5,
}

class RobotSimulation:
    def __init__(self, settings: ExperimentConfig = settings) -> None:
        self.settings = settings

        self.world: BaseWorld = WORLD_TYPES[self.settings.WORLD]()
        self.robot: CoreModule = ROBOT_TYPES[self.settings.ROBOT]()

        # --- EXPERIMENT CONSTANTS --- #
        self.SPAWN_POS: list[float] = settings.SPAWN_POS  # where the robot starts
        self.TARGET_POSITION: list[float] = settings.TARGET_POSITION  # where it should end up
        self.SIM_DURATION: float = settings.SIM_DURATION  # seconds of simulated time per evaluation
        self.MODE: ViewerTypes = settings.MODE  # see run_experiment() for the options

        # --- World and robot --------------------------------------------------- #
        self.world.spawn(
            self.robot.spec,
            position=self.SPAWN_POS,
            correct_collision_with_floor=True,
        )

        # Compile the world into a model. USE AS IS.
        self.model = self.world.spec.compile()
        self.data = mj.MjData(self.model)

        # --- Wire up the controller -------------------------------------------- #
        # Sizes are read from the compiled model, never hardcoded - they depend on
        # the body you chose in build_robot().

        # Controller architecture - decide before writing your EA.
        self.HIDDEN_SIZE: int = settings.HIDDEN_SIZE

        # output == amount of hinges of a robot, input == all input channels, i will change them
        # now. we will see
        self.INPUT_SIZE = len(self.data.qpos)
        self.OUTPUT_SIZE = self.model.nu

    def nn_controller(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        weights: list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """Map robot state to hinge commands: in -> hidden -> actions.
        Parameters
        ----------
        model : mj.MjModel
            The MuJoCo model. Use `model.nu` for the number of hinges.
        data : mj.MjData
            The MuJoCo data. This is where you read the robot's state from.
        weights : list of ndarray
            [w1, w2] - the layer weight matrices.

        Returns
        -------
        npt.NDArray[np.float64]
            `model.nu` action values, already scaled to [-pi/2, pi/2].
        """
        w1, w2 = weights

        # --- INPUTS ---------------------------------------------------------- #
        # Bare qpos - the simplest choice, not necessarily a good one.
        inputs = data.qpos

        # --- FORWARD PASS ----------------------------------------------------- #
        layer1 = np.tanh(inputs @ w1)
        outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

        # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
        return outputs * (np.pi / 2)  # in [-pi/2, pi/2]


    def get_core_position(self, data: mj.MjData) -> npt.NDArray[np.float64]:
        """Return the robot core's current (x, y, z) world position."""
        return np.asarray(data.qpos[0:3]).copy()


    def fitness_function(
        self,
        initial_position: npt.NDArray[np.float64],
        final_position: npt.NDArray[np.float64],
    ) -> float:
        """Score one evaluation. LOWER IS BETTER.

        The plain version: how far is the robot from the target when time runs out?

        `initial_position` is unused here on purpose - it is passed in because the
        moment you want a less naive fitness you will need it. Some things worth
        thinking about (and, ideally, comparing in your report):
        * Distance *reduced* rather than distance remaining, so a robot that
            starts closer is not rewarded for standing still.
        * Penalising a robot that falls over or leaves the arena.
        * Whether the z-axis should count at all - a robot that jumps is not
            closer to the target in any way you care about.
        See `ariel.simulation.tasks.targeted_locomotion` for some worked variants.
        """
        target = np.asarray(self.TARGET_POSITION)
        return float(np.linalg.norm(final_position[:2] - target[:2]))

    def run_experiment(self, ind_weights: list[npt.NDArray[np.float64]] | None = None, mode: ViewerTypes = settings.MODE) -> float:
        """Set up the world, run one simulation, and return the fitness.

        This is the function your EA calls once per individual, with `mode` set
        to "simple" (headless).

        Returns
        -------
        float
            The fitness of this run. Lower is better.
        """

        # TODO: clear it a little of the clutter

        # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
        mj.set_mjcb_control(None)

        

        # Put the simulation in a clean, known state before reading anything.
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        

        # run_experiment() will run a single experiment with 1 individual's weights or fall back to random weights
        if ind_weights is not None:
            weights = ind_weights
        else:
            weights = make_random_weights(self.INPUT_SIZE, self.OUTPUT_SIZE)


        def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
            """Compute and apply actions; MuJoCo calls this every physics step."""
            actions = self.nn_controller(m, d, weights)

            # DIRECT application (see the controller contract above).
            d.ctrl[:] = actions

            # DELTA application - comment out the line above and use these instead:
            # delta = 0.05
            # d.ctrl[:] += actions * delta
            # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

        # --- Record the starting point ----------------------------------------- #
        initial_position = self.get_core_position(self.data)

        # --- Run ---------------------------------------------------------------- #
        if mode != "no_control":
            mj.set_mjcb_control(control_callback)

        match mode:
            case "launcher":
                # Interactive window. Great for seeing what your robot does,
                # useless inside an evolutionary loop.
                viewer.launch(model=self.model, data=self.data)
            case "simple":
                # Headless. THIS is the one your EA uses.
                simple_runner(self.model, self.data, duration=self.SIM_DURATION)
            case "video":
                # Render to an mp4 - for the figures in your report.
                recorder = VideoRecorder(output_folder=str(settings.DATA / "__videos__"))
                video_renderer(
                    self.model,
                    self.data,
                    duration=settings.SIM_DURATION,
                    video_recorder=recorder,
                )
            case "frame":
                # A single image of the scene. Useful to check your spawn position
                # and that the robot is not clipping through the floor.
                single_frame_renderer(self.model, self.data, steps=1, show=True)
            case "no_control":
                # No controller attached: drag the hinges around by hand.
                viewer.launch(model=self.model, data=self.data)

        # Detach the callback again so the next run starts clean.
        mj.set_mjcb_control(None)

        # --- Score -------------------------------------------------------------- #
        final_position = self.get_core_position(self.data)
        fitness = self.fitness_function(initial_position, final_position)

        console.log(f"start  : {np.round(initial_position, 3)}")
        console.log(f"end    : {np.round(final_position, 3)}")
        console.log(f"target : {np.round(settings.TARGET_POSITION, 3)}")
        console.log(f"fitness: {fitness:.4f}   (lower is better)")

        return fitness