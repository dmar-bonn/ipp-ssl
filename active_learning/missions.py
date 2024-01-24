import os.path
from typing import Dict

import cv2
import numpy as np
import torch
from agri_semantics.utils.utils import infer_anno_and_epistemic_uncertainty_from_image
from pytorch_lightning import LightningModule

from mapper.common import Mapper
from planner.common import Planner
from planner.common import compute_flight_time
from simulator import Simulator
from utils.logger import Logger


class Mission:
    def __init__(
        self,
        planner: Planner,
        local_navigation_planner: Planner,
        mapper: Mapper,
        simulator: Simulator,
        model: LightningModule,
        init_pose: np.array,
        cfg: Dict,
        model_cfg: Dict,
        logger: Logger,
    ):
        self.logger = logger
        self.planner = planner
        self.local_navigation_planner = local_navigation_planner
        self.mapper = mapper
        self.simulator = simulator
        self.model = model
        self.init_pose = init_pose

        self.altitude = cfg["planner"]["altitude"]
        self.budget = cfg["planner"]["budget"]
        self.uav_specifications = cfg["planner"]["uav_specifications"]
        self.use_informed_map_prior = cfg["planner"]["informed_map_prior"]

        self.map_continuous_sensor_stream = cfg["mapper"]["map_continuous_sensor_stream"]

        self.simulator_name = cfg["simulator"]["name"]
        self.sensor_resolution = cfg["simulator"][self.simulator_name]["sensor"]["resolution"]
        self.sensor_angle = cfg["simulator"][self.simulator_name]["sensor"]["angle"]

        self.cfg = cfg
        self.model_cfg = model_cfg

    def compute_informed_map_prior(self, mission_id: int, store_prior_map: bool = True):
        for measurement in self.logger.all_waypoints:
            map_data = self.infer_map_data(measurement)
            self.mapper.update_map(map_data)

        for measurement in self.logger.all_train_poses:
            map_data = self.infer_map_data(measurement)
            self.mapper.update_map(map_data)

        if store_prior_map:
            file_id = f"{self.cfg['simulator']['name']}_{self.cfg['planner']['type']}_{mission_id}_prior"
            self.logger.save_maps_to_disk(
                self.mapper.terrain_map.get_prob_map(),
                self.mapper.epistemic_map.mean_map,
                file_id,
                self.cfg["mapper"]["map_name"],
            )

    def infer_map_data(self, measurement: Dict) -> Dict:
        image = measurement["image"]
        if image.shape[2] == 3:
            image = cv2.cvtColor(measurement["image"], cv2.COLOR_BGR2RGB)

        image = image.transpose(1, 0, 2)
        (
            probs,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            _,
        ) = infer_anno_and_epistemic_uncertainty_from_image(
            self.model,
            image,
            num_mc_epistemic=self.model_cfg["train"]["num_mc_epistemic"],
            resize_image=False,
            aleatoric_model=False,
            num_mc_aleatoric=-1,
            ensemble_model=False,
            evidential_model=False,
            task="classification",
        )

        _, preds = torch.max(torch.from_numpy(probs), dim=0)
        image, preds, epistemic_uncertainty, probs = (
            image.transpose(1, 0, 2),
            preds.transpose(1, 0),
            epistemic_uncertainty.transpose(1, 0),
            probs.transpose(0, 2, 1),
        )

        return {
            "pose": measurement["pose"],
            "depth": measurement["depth"],
            "logits": probs,
            "uncertainty": epistemic_uncertainty,
            "fov": measurement["fov"] if "fov" in measurement else None,
            "gsd": measurement["gsd"] if "gsd" in measurement else None,
            "is_train_data": measurement["is_train_data"],
        }

    def execute(self, mission_id: int):
        if self.use_informed_map_prior:
            self.compute_informed_map_prior(mission_id, store_prior_map=True)

        previous_pose = self.init_pose
        timestep = 0
        measurement_step = 0
        while self.budget > 0:
            measurement = self.simulator.get_measurement(previous_pose, True, mission_id)
            self.logger.add_train_pose(measurement)

            map_data = self.infer_map_data(measurement)
            self.mapper.update_map(map_data)

            if self.cfg["annotations"]["human"]["use_human_labels"]:
                self.logger.save_image_to_disk(
                    measurement["image"],
                    self.model_cfg["data"]["path_to_dataset"],
                    dataset_folder=os.path.join("training_set", "human"),
                )
                self.logger.save_qualitative_results(
                    measurement["image"],
                    measurement["anno"],
                    map_data,
                    mission_id,
                    timestep,
                    self.mapper.map_name,
                    pseudo_labels=False,
                )

            if (
                self.cfg["annotations"]["pseudo"]["use_pseudo_labels"]
                and self.cfg["annotations"]["pseudo"]["image_source"] == "waypoint"
            ):
                self.logger.save_image_to_disk(
                    measurement["image"],
                    self.model_cfg["data"]["path_to_dataset"],
                    dataset_folder=os.path.join("training_set", "pseudo"),
                )

            pose = self.planner.replan(
                self.budget,
                previous_pose,
                uncertainty_image=map_data["uncertainty"],
                mission_id=mission_id,
            )
            if pose is None:
                print(f"FINISHED '{self.planner.planner_name}' PLANNING MISSION")
                break

            if self.map_continuous_sensor_stream:
                measurement_step = self.reach_next_pose(previous_pose, pose, mission_id, measurement_step)

            self.simulator.move_to_next_waypoint(pose)
            self.budget -= compute_flight_time(pose[:3], previous_pose[:3], uav_specifications=self.uav_specifications)
            print(f"REACHED NEXT POSE: {pose}, REMAINING BUDGET: {self.budget}")

            previous_pose = pose
            timestep += 1

        file_id = f"{self.cfg['simulator']['name']}_{self.cfg['planner']['type']}_{mission_id}"
        self.logger.save_path_to_disk(file_id)
        self.logger.save_maps_to_disk(
            self.mapper.terrain_map.get_prob_map(),
            self.mapper.epistemic_map.mean_map,
            file_id,
            self.cfg["mapper"]["map_name"],
        )

    def reach_next_pose(
        self, current_pose: np.array, next_pose: np.array, mission_id: int, measurement_step: int
    ) -> int:
        local_path = self.local_navigation_planner.replan(
            self.budget,
            current_pose,
            next_pose=next_pose,
            sensor_frequency=self.cfg["simulator"][self.simulator_name]["sensor"]["frequency"],
            subsample_path=True,
        )

        for pose in local_path:
            self.simulator.move_to_next_waypoint(pose)
            measurement = self.simulator.get_measurement(pose, False, mission_id)
            map_data = self.infer_map_data(measurement)
            self.mapper.update_map(map_data)
            self.logger.add_waypoint(pose, measurement)
            if (
                self.cfg["annotations"]["pseudo"]["use_pseudo_labels"]
                and self.cfg["annotations"]["pseudo"]["image_source"] == "in_between"
            ):
                self.logger.save_image_to_disk(
                    measurement["image"],
                    self.model_cfg["data"]["path_to_dataset"],
                    dataset_folder=os.path.join("training_set", "pseudo"),
                )
                self.logger.save_qualitative_results(
                    measurement["image"],
                    measurement["anno"],
                    map_data,
                    mission_id,
                    measurement_step,
                    self.mapper.map_name,
                    pseudo_labels=True,
                )
                measurement_step += 1

        return measurement_step
