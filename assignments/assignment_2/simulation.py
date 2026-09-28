from pathlib import Path

from settings import settings, ExperimentConfig
from assignment_types import Coordinates, FloatArray, Position, ViewerTypes, Weights
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
from mujoco import viewer

# moduels
from factories import FitnessFactory

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

TARGET_POSITIONS: dict[str, Coordinates] = {
    "SimpleFlatWorld": (2.0, 0.0, 0.1),
    "RuggedTerrainWorld": (2.0, 0.0, 0.1),
    "CraterTerrainWorld": (2.0, 0.0, 0.1),
    "AmphitheatreTerrainWorld": (3.5, 0.0, 0.1),
    "OlympicArena": (5.42, 0.0, 0.506),
}

class RobotSimulation:
    def __init__(
        self,
        settings: ExperimentConfig = settings,
        *,
        output_folder: Path | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.settings = settings
        self._fitness = FitnessFactory(
            settings.FITNESS_TYPE, fall_penalty=settings.FALL_PENALTY,
            speed_penalty=settings.SPEED_PENALTY,
        )
        self.output_folder = output_folder if output_folder is not None else settings.DATA
        self.rng = rng if rng is not None else np.random.default_rng()
        self.total_path_length = 0.0
        self.mean_target_distance = 0.0

        self.world: BaseWorld = WORLD_TYPES[self.settings.WORLD]()
        self.robot: CoreModule = ROBOT_TYPES[self.settings.ROBOT]()

        # --- EXPERIMENT CONSTANTS --- #
        self.SPAWN_POS: list[float] = settings.SPAWN_POS  # where the robot starts
        self.TARGET_POSITION: Coordinates = TARGET_POSITIONS[self.settings.WORLD]  # where it should end up
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
        # actually i might be wrong about input
        # now. we will see
        # Two extra channels give the controller a repeating sine/cosine clock.
        self.INPUT_SIZE = len(self.data.qpos) + len(self.data.qvel) + 2
        self.OUTPUT_SIZE = self.model.nu

        # check geoms for understanding how to interact with the arena
        # collect arena boundaries to understand the runs where the robot is falling from the arena
        
        # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
        # without these 2 lines all boundaries read will be 0 0 0 
        mj.set_mjcb_control(None)
        mj.mj_forward(self.model, self.data)

        self.boundaries = {}
        for i in range(self.model.ngeom):
            # check only the geoms of the arena itself not the robot
            if not self.model.geom(i).name.split("_")[0].startswith("robot"):

                # get the geom and the state from mjdata
                geom = self.model.geom(i)
                state = self.data.geom(i)
                print("geom", geom)
                print("state", state)
                # understand its rotation to be able to map into world coordingates
                rotation = state.xmat.reshape(3, 3)
                print("rotation", rotation)

                # understand the local center and halfsize to understand the span of the bb along the axis
                local_center, half_size = self.model.geom_aabb[i].reshape(2, 3)
                print("local_center", local_center, "half_size", half_size)

                # understand actual world coordinate center applying rotation and local center
                center = state.xpos + rotation @ local_center
                # translate to world halfsize an duse absolute values to not cancel out the distances
                world_half_size = np.abs(rotation) @ half_size
                print("center",center)
                print("world_half_size",world_half_size)

                # calculate the coords using worlds half size
                minxyz = center - world_half_size
                maxxyz = center + world_half_size

                print(
                    geom.name,
                    "min XYZ:", minxyz, 
                    "max XYZ:", maxxyz,
                )

                self.boundaries[geom.name] = (minxyz, maxxyz)

    def nn_controller(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        weights: Weights,
    ) -> FloatArray:
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
        FloatArray
            `model.nu` action values, already scaled to [-pi/2, pi/2].
        """
        w1, w2 = weights

        # --- INPUTS ---------------------------------------------------------- #
        # Simulation time resets each evaluation, so the rhythm does too.
        phase = 2 * np.pi * self.settings.RHYTHM_FREQUENCY * data.time
        rhythm = np.array([np.sin(phase), np.cos(phase)])
        inputs = np.concatenate((data.qpos, data.qvel, rhythm))

        # --- FORWARD PASS ----------------------------------------------------- #
        layer1 = np.tanh(inputs @ w1)
        outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

        # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
        return outputs * (np.pi / 2)  # in [-pi/2, pi/2]


    def get_core_position(self, data: mj.MjData) -> Position:
        """Return the robot core's current (x, y, z) world position."""
        return np.asarray(data.qpos[0:3]).copy()


    def fitness_function(
        self,
        initial_position: Position,
        final_position: Position,
        total_path_length: float,
    ) -> float:
        """Score one evaluation using the configured fitness. Lower is better."""
        return self._fitness(
            initial_position,
            final_position,
            np.asarray(self.TARGET_POSITION, dtype=np.float64),
            self.if_onplatform(final_position) if self.settings.FITNESS_TYPE == "improved" else True,
            total_path_length,
            self.mean_target_distance,
        )

    # nothingburger engineering
    # def which_onplatform(self, robot_pos: Position) -> str:
    #     for pl in self.boundaries.keys():
    #         mn, mx = self.boundaries[pl]
    #         if np.all(mn[:2] < robot_pos[:2]) and np.all(mx[:2]> robot_pos[:2]):
    #             return pl
    #     # return "" if the robot is not within any platform
    #     return ""

    def if_onplatform(self, robot_pos: Position) -> bool:
        for pl in self.boundaries.keys():
            # check boundaries
            mn, mx = self.boundaries[pl]
            # only compare X and Y here because for Z the rules of determinng if robot is on the platform are differnet
            if np.all(mn[:2] < robot_pos[:2]) and np.all(mx[:2]> robot_pos[:2]):
                # get rotation because we need to see if platform is tiltedd
                rot = self.get_geom_rotation(pl)
                # no rotation will equal identity matrix
                # but it doesnt always mean that if platform is not rotated,
                # its xmat values will be exactly np.eye(3)
                # imagine this:
                # [[ 0.707, -0.707, 0 ]
                # [ 0.707, 0.707, 0 ]
                # [ 0, 0, 1 ]]
                # the platform is only rotated around Z which is still flat
                # additionally
                # the actual rotation of the non tilted platforms is 
                # [[ 0, 0, 1 ]
                # [ 0, -1, 0]
                # [1, 0, 0]]
                # which means that its local X and Z are swapped and Y is -Y
                # compared to world coordinates
                # so we only care about local Z and Y rotation
                # only the world Z rotation would not change the relationship between world Z and local X
                # which is the rot [2, 0]
                # and we only care about it being flatso 
                # so what we do is simple
                # upd: i realized im stupid this was not needed. i didnt need it at all bro
                is_flat = np.isclose(abs(rot[2, 0]), 1.0)
                # we check if the platforms 
                if is_flat:
                    if mx[2] <= robot_pos[2]:
                        return True
                else:
                    if mn[2] < robot_pos[2]:

                        return True

        return False

    def get_geom_rotation(self, name: str) -> np.ndarray: # TODO: i need a type denoting rotation asap

        mj.mj_forward(self.model, self.data)

        try:
            for i in range(self.model.ngeom):
                if name == self.model.geom(i).name:
                    # reshape so its a 3x3 matrix
                    return self.data.geom(i).xmat.reshape(3, 3).copy()
        except Exception as e:
            raise KeyError(e, "no geom with this name found")
        

    def run_experiment(self, ind_weights: Weights | None = None, mode: ViewerTypes | None = None) -> float:
        """Set up the world, run one simulation, and return the fitness.

        This is the function your EA calls once per individual, with `mode` set
        to "simple" (headless).

        Returns
        -------
        float
            The fitness of this run. Lower is better.
        """

        if mode is None:
            mode = self.MODE

        # TODO: clear it a little of the clutter

        # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
        mj.set_mjcb_control(None)

        # Put the simulation in a clean, known state before reading anything.
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        # run_experiment() will run a single experiment with 1 individual's weights or fall back to random weights
        weights = ind_weights

        initial_position = self.get_core_position(self.data)
        previous_xy = initial_position[:2].copy()
        previous_time = float(self.data.time)
        initial_time = previous_time
        target_xy = np.asarray(self.TARGET_POSITION[:2])
        previous_distance = float(np.linalg.norm(previous_xy - target_xy))
        distance_integral = 0.0
        self.mean_target_distance = previous_distance
        self.total_path_length = 0.0

        def record_path(d: mj.MjData) -> None:
            nonlocal previous_xy, previous_time, previous_distance, distance_integral
            # Rendering/forward calls can invoke the callback again at the same time.
            if d.time <= previous_time:
                return
            current_xy = d.qpos[:2].copy()
            self.total_path_length += float(np.linalg.norm(current_xy - previous_xy))
            current_distance = float(np.linalg.norm(current_xy - target_xy))
            distance_integral += 0.5 * (previous_distance + current_distance) * (d.time - previous_time)
            previous_distance = current_distance
            previous_xy = current_xy
            previous_time = float(d.time)
            self.mean_target_distance = distance_integral / (previous_time - initial_time)

        def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
            """Compute and apply actions; MuJoCo calls this every physics step."""
            record_path(d)
            if mode == "no_control":
                return
            actions = self.nn_controller(m, d, weights)

            # DIRECT application (see the controller contract above).
            d.ctrl[:] = actions

            # DELTA application - comment out the line above and use these instead:
            # delta = 0.05
            # d.ctrl[:] += actions * delta
            # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

        # Observe movement in every mode, including manual control.
        mj.set_mjcb_control(control_callback)

        try:
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
                    recorder = VideoRecorder(output_folder=str(self.output_folder / "__videos__"))
                    video_renderer(
                        self.model,
                        self.data,
                        duration=self.SIM_DURATION,
                        video_recorder=recorder,
                    )
                case "frame":
                    # A single image of the scene. Useful to check your spawn position
                    # and that the robot is not clipping through the floor.
                    single_frame_renderer(self.model, self.data, steps=1, show=True)
                case "no_control":
                    # No controller attached: drag the hinges around by hand.
                    viewer.launch(model=self.model, data=self.data)
        finally:
            mj.set_mjcb_control(None)

        # --- Score -------------------------------------------------------------- #
        # Include the final step, which occurs after the last controller callback.
        record_path(self.data)
        final_position = self.get_core_position(self.data)
        fitness = self.fitness_function(
            initial_position, final_position, self.total_path_length,
        )

        console.log(f"start  : {np.round(initial_position, 3)}")
        console.log(f"end    : {np.round(final_position, 3)}")
        console.log(f"target : {np.round(self.TARGET_POSITION, 3)}")
        console.log(f"path XY: {self.total_path_length:.4f}")
        console.log(f"mean target distance: {self.mean_target_distance:.4f}")
        console.log(f"fitness: {fitness:.4f}   (lower is better)")

        return fitness