from typing import Dict

import cv2
import numpy as np

from simulator import Sensor, Simulator
from utils import utils


class RGBSensor(Sensor):
    def __init__(self, cfg: Dict):
        super(RGBSensor, self).__init__(cfg)


class OrthoSimulator(Simulator):
    def __init__(self, cfg: Dict, world: np.array, anno: np.array):
        super(OrthoSimulator, self).__init__(cfg)

        self.simulator_name = "ortho-simulator"
        self.world = world
        self.anno = anno

    def setup_sensor(self, sensor_cfg: Dict) -> RGBSensor:
        return RGBSensor(sensor_cfg)

    def get_measurement(self, pose: np.array, is_train_data: bool, mission_id: int) -> Dict:
        fov_info = utils.get_fov(pose, self.sensor.angle, self.gsd, self.world_range)

        fov_corner, range_list = fov_info
        gsd = [
            (np.linalg.norm(fov_corner[1] - fov_corner[0])) / self.sensor.resolution[0],
            (np.linalg.norm(fov_corner[3] - fov_corner[0])) / self.sensor.resolution[1],
        ]
        rgb_image_raw = self.world[range_list[2] : range_list[3], range_list[0] : range_list[1], :]
        rgb_image = cv2.resize(rgb_image_raw, tuple(self.sensor.resolution))
        depth_image = None

        return {
            "image": rgb_image,
            "depth": depth_image,
            "anno": self.get_anno(pose),
            "fov": fov_corner,
            "range": range_list,
            "gsd": gsd,
            "is_train_data": is_train_data,
            "mission_id": mission_id,
            "pose": pose,
        }

    def get_anno(self, pose: np.array) -> np.array:
        _, range_list = utils.get_fov(pose, self.sensor.angle, self.gsd, self.world_range)

        if len(self.anno.shape) == 3:
            rgb_anno_raw = self.anno[range_list[2] : range_list[3], range_list[0] : range_list[1], :]
        else:
            rgb_anno_raw = self.anno[range_list[2] : range_list[3], range_list[0] : range_list[1]]

        return cv2.resize(rgb_anno_raw, tuple(self.sensor.resolution))
