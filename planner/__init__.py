from typing import Dict

from mapper.common import Mapper
from planner.baselines import CoveragePlanner
from planner.common import Planner
from planner.frontier import FrontierPlanner
from planner.local_navigation import LocalNavigationPlanner


def get_local_navigation_planner(cfg: Dict, mapper: Mapper) -> LocalNavigationPlanner:
    simulator_name = cfg["simulator"]["name"]
    planner = LocalNavigationPlanner(
        mapper,
        cfg["planner"]["altitude"],
        cfg["simulator"][simulator_name]["sensor"],
        cfg["planner"]["uav_specifications"],
        cfg["planner"]["objective_fn"],
    )
    planner.setup()
    return planner


def get_planner(cfg: Dict, mapper: Mapper, **kwargs) -> Planner:
    simulator_name = cfg["simulator"]["name"]
    planner_type = cfg["planner"]["type"]
    planner_params = cfg["planner"][planner_type]
    if planner_type == "coverage":
        planner = CoveragePlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"],
            cfg["planner"]["uav_specifications"],
            planner_params["step_sizes"],
            cfg["planner"]["objective_fn"],
        )
        planner.setup(mission_id=kwargs["mission_id"], starting_position=cfg["planner"]["starting_position"])
        return planner
    elif planner_type == "frontier":
        planner = FrontierPlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"],
            cfg["planner"]["uav_specifications"],
            planner_params["step_size"],
            planner_params["sensor_angle"],
            planner_params["sensor_resolution"],
            cfg["planner"]["objective_fn"],
        )
        planner.setup()
        return planner
    else:
        raise ValueError(f"Planner type '{planner_type}' unknown!")
