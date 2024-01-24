from typing import Dict, List, Optional

import numpy as np

from mapper.common import Mapper
from planner.common import Planner


class LocalNavigationPlanner(Planner):
    def __init__(
        self,
        mapper: Mapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        objective_fn_name: str,
    ):
        super(LocalNavigationPlanner, self).__init__(
            mapper,
            altitude,
            sensor_info,
            uav_specifications,
            objective_fn_name,
        )

        self.planner_name = "straight-line"
        self.map_resolution = self.mapper.ground_resolution
        self.map_boundary = self.mapper.map_boundary
        if self.mapper.use_torch:
            self.map_resolution = self.mapper.ground_resolution.cpu().numpy()
            self.map_boundary = self.mapper.map_boundary.cpu().numpy()

        self.map_boundary = self.map_boundary.astype(int)

    def extend_fn(self, start_pose: np.array, end_pose: np.array) -> List:
        direction = end_pose - start_pose
        distance = np.linalg.norm(direction, ord=2)
        sample_dist = 0.2 * self.map_resolution[0]
        sample_steps = np.arange(0, distance, sample_dist)
        return [start_pose + sample_step * direction / distance for sample_step in sample_steps]

    def compute_straight_line(self, start_pose: np.array, end_pose: np.array) -> List:
        return self.extend_fn(start_pose[:2], end_pose[:2])

    def complete_path(self, path: List) -> List:
        completed_path = []
        for i, xy_pos in enumerate(path):
            completed_path.append(np.array([xy_pos[0], xy_pos[1], self.altitude]))

        return completed_path

    def subsample_path(self, local_path: List, sensor_frequency: float) -> List:
        sample_dist = self.uav_specifications["max_v"] / sensor_frequency
        dist_integrated = 0.0
        subsampled_path = []

        for i in range(1, len(local_path)):
            dist_integrated += np.linalg.norm(local_path[i - 1][:3] - local_path[i][:3], ord=2)
            if dist_integrated >= sample_dist:
                subsampled_path.append(local_path[i])
                dist_integrated = 0.0

        return subsampled_path

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[List]:
        local_path = self.compute_straight_line(previous_pose, kwargs["next_pose"])
        local_path = self.complete_path(local_path)

        if kwargs["subsample_path"]:
            local_path = self.subsample_path(local_path, kwargs["sensor_frequency"])

        return local_path
