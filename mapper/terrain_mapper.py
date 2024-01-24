from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from mapper.common import ContinuousVariableMap, CountMap, DiscreteVariableMap, Mapper
from simulator import Sensor
from utils import utils


class TerrainMapper(Mapper):
    def __init__(self, cfg: Dict, model_cfg: Dict, simulator_name: str, sensor: Sensor):
        super(TerrainMapper, self).__init__(cfg, model_cfg, simulator_name, sensor, use_torch=False)

        self.gsd = cfg["simulator"][simulator_name]["gsd"]

        (
            self.terrain_map,
            self.epistemic_map,
            self.occupancy_map,
            self.hit_map_occupancy,
            self.train_data_map,
            _,
        ) = self.init_map()

    def init_map(
        self,
    ) -> Tuple[
        Union[ContinuousVariableMap, DiscreteVariableMap],
        ContinuousVariableMap,
        ContinuousVariableMap,
        CountMap,
        CountMap,
        Optional[CountMap],
    ]:
        hit_map = CountMap(self.cfg)
        epistemic_map = ContinuousVariableMap(self.cfg, self.model_cfg, hit_map, "mle", 1)
        occupancy_map = ContinuousVariableMap(self.cfg, self.model_cfg, hit_map, "mle", 1, occupancy_map=True)

        task = self.cfg["simulator"]["task"]
        if task == "regression":
            terrain_map = ContinuousVariableMap(self.cfg, self.model_cfg, hit_map, "map", 1)
        elif task == "classification":
            terrain_map = DiscreteVariableMap(self.cfg, self.model_cfg, self.class_num)
        else:
            raise NotImplementedError(f"Semantic map update for '{task}' task not implemented!")

        return terrain_map, epistemic_map, occupancy_map, hit_map, CountMap(self.cfg), None

    def find_map_index(self, data_point) -> Tuple[float, float]:
        x_index = np.floor(data_point[0] / self.ground_resolution[0]).astype(int)
        y_index = np.floor(data_point[1] / self.ground_resolution[1]).astype(int)

        return x_index, y_index

    def update_map(self, data_source: Dict):
        semantics = data_source["logits"]
        uncertainty = data_source["uncertainty"]
        fov = data_source["fov"]
        gsd = data_source["gsd"]
        is_train_data = data_source["is_train_data"]
        _, m_y_dim, m_x_dim = semantics.shape

        measurement_indices = np.array(np.meshgrid(np.arange(m_y_dim), np.arange(m_x_dim))).T.reshape(-1, 2).astype(int)
        x_ground = fov[0][0] + (0.5 + np.arange(m_x_dim)) * gsd[0]
        y_ground = fov[0][1] + (0.5 + np.arange(m_y_dim)) * gsd[1]
        ground_coords = np.array(np.meshgrid(y_ground, x_ground)).T.reshape(-1, 2)
        map_indices = np.floor(ground_coords / np.array(self.ground_resolution)).astype(int)

        semantics_proj = semantics[:, measurement_indices[:, 0], measurement_indices[:, 1]]
        uncertainty_proj = uncertainty[measurement_indices[:, 0], measurement_indices[:, 1]]

        self.hit_map_occupancy.update(map_indices)
        if is_train_data:
            self.train_data_map.update(map_indices)

        self.occupancy_map.update(map_indices, np.ones_like(uncertainty_proj))
        self.epistemic_map.update(map_indices, uncertainty_proj, variance_measured=None)
        self.terrain_map.update(map_indices, semantics_proj, variance_measured=uncertainty_proj)

    def get_map_state(
        self, pose: np.array, depth: np.array = None, sensor: Sensor = None
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        if sensor is None:
            sensor = self.sensor

        fov_corners, _ = utils.get_fov(pose[:3], sensor.angle, self.ground_resolution[0], self.map_boundary)
        lu, _, rd, _ = fov_corners
        lu_x, lu_y = self.find_map_index(lu)
        rd_x, rd_y = self.find_map_index(rd)

        semantic_map_probs = (
            self.terrain_map.get_prob_map()[:, lu_y:rd_y, lu_x:rd_x].transpose(1, 2, 0).astype(np.float32)
        )
        semantic_map_probs = cv2.resize(semantic_map_probs, sensor.resolution).transpose(2, 0, 1)

        map_uncertainties = self.epistemic_map.mean_map[0, lu_y:rd_y, lu_x:rd_x].astype(np.float32)
        map_uncertainties = cv2.resize(map_uncertainties, sensor.resolution)

        hit_count_map = self.hit_map_occupancy.count_map[lu_y:rd_y, lu_x:rd_x].copy()
        unknown_space_map = cv2.resize(hit_count_map, sensor.resolution) == 0

        train_data_count_map = self.train_data_map.count_map[lu_y:rd_y, lu_x:rd_x].copy()
        train_data_count_map = cv2.resize(train_data_count_map, sensor.resolution)

        return semantic_map_probs, map_uncertainties, ~unknown_space_map, unknown_space_map, train_data_count_map
