from typing import Dict, Tuple

import numpy as np


class Sensor:
    def __init__(self, cfg: Dict):
        self.angle = np.array(cfg["angle"])
        self.resolution = np.array(cfg["resolution"])
        self.depth_noise = 0.2
        self.orientation = "downward"
        if "orientation" in cfg.keys():
            self.orientation = cfg["orientation"]

        self.min_depth, self.max_depth = 0.2, 25.0

    @property
    def focal_length(self) -> Tuple[float, float]:
        fx = fy = (self.resolution[1] / 2.0) / np.tan(np.pi * self.angle[0] / 180.0)
        return fx, fy

    @property
    def principal_point(self) -> Tuple[float, float]:
        return self.resolution[1] / 2.0, self.resolution[0] / 2.0


class Simulator:
    def __init__(self, cfg: Dict):
        self.simulator_name = "base-simulator"

        self.gsd = cfg["gsd"]  # m/pixel
        self.world_range = cfg["world_range"]  # pixel
        self.sensor = self.setup_sensor(cfg["sensor"])

    def setup_sensor(self, sensor_cfg: Dict) -> Sensor:
        raise NotImplementedError(f"Simulator '{self.simulator_name}' does not implement 'setup_sensor()' function!")

    def get_measurement(self, pose: np.array, is_train_data: bool, mission_id: int) -> Dict:
        raise NotImplementedError(f"Simulator '{self.simulator_name}' does not implement 'get_measurement()' function!")

    def start_mission(self, init_pose: np.array):
        pass

    def move_to_next_waypoint(self, pose: np.array):
        pass
