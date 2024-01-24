from typing import List, Optional, Dict

import numpy as np

from mapper.common import Mapper
from planner.common import compute_flight_time, Planner


class CoveragePlanner(Planner):
    def __init__(
        self,
        mapper: Mapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        step_sizes: List[float],
        objective_fn_name: str,
    ):
        super(CoveragePlanner, self).__init__(mapper, altitude, sensor_info, uav_specifications, objective_fn_name)

        self.planner_name = "coverage-based"
        self.step_sizes = step_sizes
        self.waypoints = []
        self.step_counter = 0
        self.step_size = step_sizes[0]

    def setup(self, **kwargs):
        mission_id = kwargs["mission_id"]
        starting_position = kwargs["starting_position"]

        self.step_size = self.step_sizes[mission_id % len(self.step_sizes)]
        self.waypoints = self.create_coverage_pattern(mission_id % 2, starting_position)

    def create_coverage_pattern(self, flip_orientation: bool, starting_position: str) -> np.array:
        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle)) + 0.1
        min_y, min_x = boundary_space[1], boundary_space[0]
        max_y = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1]
        max_x = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0]

        x_positions = np.linspace(min_x, max_x, int((max_x - min_x) / self.step_size) + 2)
        y_positions = np.linspace(min_y, max_y, int((max_y - min_y) / self.step_size) + 2)

        if starting_position in ["top_right", "bottom_right"]:
            x_positions = np.flip(x_positions)

        if starting_position in ["bottom_left", "bottom_right"]:
            y_positions = np.flip(y_positions)

        waypoints = np.zeros((len(y_positions) * len(x_positions), 3))

        if flip_orientation:
            for j, x_pos in enumerate(x_positions):
                for k, y_pos in enumerate(y_positions):
                    if j % 2 == 1:
                        y_pos = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - y_pos

                    waypoints[j * len(y_positions) + k] = np.array([x_pos, y_pos, self.altitude], dtype=np.float32)
        else:
            for j, y_pos in enumerate(y_positions):
                for k, x_pos in enumerate(x_positions):
                    if j % 2 == 1:
                        x_pos = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - x_pos

                    waypoints[j * len(x_positions) + k] = np.array([x_pos, y_pos, self.altitude], dtype=np.float32)

        return waypoints

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        if self.step_counter >= len(self.waypoints):
            return None

        pose = self.waypoints[self.step_counter, :]
        self.step_counter += 1

        if compute_flight_time(pose, previous_pose, self.uav_specifications) > budget:
            return None

        return pose
