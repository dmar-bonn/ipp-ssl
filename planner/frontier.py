import copy
from typing import Dict, List, Optional

import cv2
import numpy as np

from mapper.common import Mapper
from planner.common import compute_flight_time, Planner
from simulator import Sensor


class FrontierPlanner(Planner):
    def __init__(
        self,
        mapper: Mapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        frontier_step_size: float,
        sensor_angle: List,
        sensor_resolution: List,
        objective_fn_name: str,
    ):
        super(FrontierPlanner, self).__init__(mapper, altitude, sensor_info, uav_specifications, objective_fn_name)

        self.planner_name = "frontier-based"
        self.frontier_step_size = frontier_step_size

        self.sensor = self.create_sensor(sensor_resolution, sensor_angle)
        self.map_resolution, self.map_boundary = self.mapper.ground_resolution, self.mapper.map_boundary
        if self.mapper.use_torch:
            self.map_resolution, self.map_boundary = self.map_resolution.cpu().numpy(), self.map_boundary.cpu().numpy()

    def create_sensor(self, resolution: np.array, angle: np.array) -> Sensor:
        sensor = copy.deepcopy(self.mapper.sensor)
        sensor.resolution = resolution
        sensor.angle = angle
        return sensor

    def objective_fn(self, candidate_pose: np.array) -> float:
        (
            _,
            uncertainty_image,
            occupied_space_image,
            unknown_space_image,
            train_data_count_image,
        ) = self.mapper.get_map_state(candidate_pose, depth=None, sensor=self.sensor)
        free_space_image = ~unknown_space_image & ~occupied_space_image
        uncertainty_image[free_space_image] = 0
        uncertainty_image[unknown_space_image] = self.mapper.uncertainty_prior_const
        uncertainty_image[occupied_space_image & (train_data_count_image > 0)] /= train_data_count_image[
            occupied_space_image & (train_data_count_image > 0)
        ]
        return np.sum(uncertainty_image)

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        if budget < 5.0:
            return None

        if self.mapper.terrain_map.mle_map_outdated:
            self.mapper.terrain_map.set_mle_map()
            self.mapper.terrain_map.mle_map_outdated = False

        map_resolution, map_boundary = self.mapper.ground_resolution, self.mapper.map_boundary
        if self.mapper.use_torch:
            map_resolution, map_boundary = map_resolution.cpu().numpy(), map_boundary.cpu().numpy()

        boundary_space = self.altitude * np.tan(np.deg2rad(self.mapper.sensor.angle)) + 1
        max_y = map_boundary[1] * map_resolution[1] - boundary_space[1]
        max_x = map_boundary[0] * map_resolution[0] - boundary_space[0]

        known_space_map = self.mapper.occupancy_map.mean_map[0] > 0.6
        known_space_map = known_space_map.astype(np.uint8)
        frontiers, _ = cv2.findContours(known_space_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        best_poses = [previous_pose[:3]]
        best_frontier_value = -np.inf
        for i, frontier in enumerate(frontiers):
            frontier_pose_candidates = frontier[:, 0][:: self.frontier_step_size, :]
            for j, frontier_pose_candidate in enumerate(frontier_pose_candidates):
                frontier_pose_candidate = frontier_pose_candidate * map_resolution[:2]
                frontier_pose_candidate = np.append(frontier_pose_candidate, self.altitude)

                if frontier_pose_candidate[0] < boundary_space[0] or frontier_pose_candidate[0] > max_x:
                    continue

                if frontier_pose_candidate[1] < boundary_space[1] or frontier_pose_candidate[1] > max_y:
                    continue

                if np.allclose(previous_pose[:3], frontier_pose_candidate):
                    continue

                if compute_flight_time(previous_pose[:3], frontier_pose_candidate, self.uav_specifications) > budget:
                    continue

                frontier_candidate_value = self.objective_fn(frontier_pose_candidate)

                if frontier_candidate_value >= best_frontier_value:
                    if frontier_candidate_value == best_frontier_value:
                        best_poses.append(frontier_pose_candidate)
                    else:
                        best_poses = [frontier_pose_candidate]
                    best_frontier_value = frontier_candidate_value

        sampled_best_pose = best_poses[np.random.choice(len(best_poses))]
        if np.allclose(previous_pose[:3], sampled_best_pose):
            return None

        if compute_flight_time(sampled_best_pose, previous_pose[:3], self.uav_specifications) > budget:
            return None

        return sampled_best_pose
