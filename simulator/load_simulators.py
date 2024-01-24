import os
from typing import Dict

import cv2
import numpy as np

from simulator import Simulator
from simulator.ortho_simulator import OrthoSimulator


def get_simulator(cfg: Dict) -> Simulator:
    simulator_cfg = cfg["simulator"]

    if isinstance(simulator_cfg, dict):
        print("creating simulation world")
        if simulator_cfg["name"] == "potsdam":
            potsdam_world = get_potsdam_world(simulator_cfg["potsdam"])
            cv2.imwrite("potsdam_world.png", potsdam_world)
            return OrthoSimulator(simulator_cfg["potsdam"], potsdam_world, get_anno(cfg))
        else:
            raise NotImplementedError(f"{type(simulator_cfg)} simulator not implemented!")
    else:
        raise RuntimeError(f"{simulator_cfg['name']} not a valid config file!")


def get_anno(cfg: Dict) -> np.array:
    simulator_cfg = cfg["simulator"]

    if isinstance(simulator_cfg, dict):
        print("creating oracle annotations")
        if simulator_cfg["name"] == "potsdam":
            if simulator_cfg["task"] == "classification":
                potsdam_anno = get_potsdam_semantic_segmentation(simulator_cfg["potsdam"])
            else:
                raise NotImplementedError(f'{simulator_cfg["name"]} for {simulator_cfg["task"]} not implemented!')

            cv2.imwrite(f'potsdam_{simulator_cfg["task"]}.png', potsdam_anno)
            return potsdam_anno
        else:
            raise NotImplementedError(f"{type(simulator_cfg)} simulator not implemented!")
    else:
        raise RuntimeError(f"{type(simulator_cfg)} not a valid config file")


def get_potsdam_world(cfg: Dict) -> np.array:
    path_to_orthomosaic = cfg["path_to_orthomosaic"]
    resize_flag = cfg["resize_flag"]
    resize_factor = cfg["resize_factor"]
    ortho_tile_list = cfg["ortho_tile_list"]
    ortho_rgb = []

    for row in ortho_tile_list:
        orth_rgb_row = []
        for orth_num in row:
            path_to_tile = f"{path_to_orthomosaic}/potsdam_{orth_num}_RGB.tif"
            if not os.path.exists(path_to_tile):
                raise FileNotFoundError

            rgb = cv2.imread(path_to_tile)
            if resize_flag:
                orig_height, orig_width, _ = rgb.shape
                resized_height, resized_width = int(orig_height / resize_factor), int(orig_width / resize_factor)
                rgb = cv2.resize(rgb, (resized_height, resized_width))

            orth_rgb_row.append(rgb)
        ortho_rgb.append(np.concatenate(orth_rgb_row, axis=1))

    orthomosaic = np.concatenate(ortho_rgb, axis=0)

    return orthomosaic


def get_potsdam_semantic_segmentation(cfg: Dict) -> np.array:
    path_to_orthomosaic = cfg["path_to_anno"]
    resize_flag = cfg["resize_flag"]
    resize_factor = cfg["resize_factor"]
    ortho_tile_list = cfg["ortho_tile_list"]
    ortho_anno = []

    for row in ortho_tile_list:
        ortho_anno_row = []
        for orth_num in row:
            path_to_tile = f"{path_to_orthomosaic}/potsdam_{orth_num}_label.tif"
            if not os.path.exists(path_to_tile):
                raise FileNotFoundError(f"Cannot find ortho tile '{path_to_tile}'")

            anno = cv2.imread(path_to_tile)
            if resize_flag:
                orig_height, orig_width, _ = anno.shape
                resized_height, resized_width = int(orig_height / resize_factor), int(orig_width / resize_factor)
                anno = cv2.resize(anno, (resized_height, resized_width))
            ortho_anno_row.append(anno)
        ortho_anno.append(np.concatenate(ortho_anno_row, axis=1))

    return np.concatenate(ortho_anno, axis=0)
