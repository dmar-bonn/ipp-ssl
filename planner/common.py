from typing import Dict, Optional

import numpy as np

from mapper.common import Mapper


def compute_flight_time(action: np.array, previous_action: np.array, uav_specifications: Dict = None) -> float:
    dist_total = np.linalg.norm(action - previous_action, ord=2)
    dist_acc = min(dist_total * 0.5, np.square(uav_specifications["max_v"]) / (2 * uav_specifications["max_a"]))
    dist_const = dist_total - 2 * dist_acc

    time_acc = np.sqrt(2 * dist_acc / uav_specifications["max_a"])
    time_const = dist_const / uav_specifications["max_v"]
    time_total = time_const + 2 * time_acc

    return time_total


class Planner:
    def __init__(
        self,
        mapper: Mapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        objective_fn_name: str,
    ):
        self.planner_name = "planner"
        self.mapper = mapper
        self.altitude = altitude
        self.sensor_angle = sensor_info["angle"]
        self.sensor_resolution = sensor_info["resolution"]
        self.uav_specifications = uav_specifications
        self.objective_fn_name = objective_fn_name

    def get_schematic_image(
        self, uncertainty_image: np.array, representation_image: np.array, mission_id: int
    ) -> np.array:
        if self.objective_fn_name in ["epistemic_uncertainty", "aleatoric_uncertainty", "predictive_uncertainty"]:
            return uncertainty_image
        elif self.objective_fn_name == "hybrid":
            return uncertainty_image if mission_id < 5 else representation_image
        elif self.objective_fn_name == "representation":
            return representation_image
        else:
            raise NotImplementedError(
                f"Planning objective function '{self.objective_fn_name}' not implemented for {self.planner_name} planner"
            )

    def setup(self, **kwargs):
        pass

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        raise NotImplementedError("Replan function not implemented!")
