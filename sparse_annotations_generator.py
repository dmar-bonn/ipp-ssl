import os
from typing import Dict, Tuple

import click
import numpy as np
import yaml

from simulator.load_simulators import get_simulator
from utils.logger import Logger


def read_config_files(config_file_path: str) -> Tuple[Dict, Dict]:
    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"Cannot find config file '{config_file_path}'!")

    if not config_file_path.endswith((".yaml", ".yml")):
        raise ValueError(f"Config file is not a yaml-file! Only '.yaml' or '.yaml' file endings allowed!")

    with open(config_file_path, "r") as file:
        cfg = yaml.safe_load(file)

    with open(cfg["network"]["path_to_config"], "r") as config_file:
        model_cfg = yaml.safe_load(config_file)

    return cfg, model_cfg


def sample_annotation_mask(sensor_resolution: np.array, num_labelled_pixels: int) -> np.array:
    anno_mask = np.zeros(np.prod(sensor_resolution), dtype=bool)
    anno_mask[:num_labelled_pixels] = 1
    anno_mask = anno_mask.reshape([sensor_resolution[1], sensor_resolution[0]])
    np.random.shuffle(anno_mask)
    return anno_mask.astype(bool)


@click.command()
@click.option(
    "--config_file",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml"),
)
@click.option(
    "--dataset_folder",
    "-d",
    type=str,
    help="dataset folder name, either 'training_set', 'validation_set' or 'test_set'",
    default="training_set",
)
@click.option(
    "--num_data_samples", "-n", type=int, help="number of to-be-generated image-annotation data points", default=1000
)
@click.option("--num_pixels_per_sample", "-m", type=int, help="number of to-be-labelled pixels per images", default=100)
def main(config_file: str, dataset_folder: str, num_data_samples: int, num_pixels_per_sample: int):
    cfg, model_cfg = read_config_files(config_file)

    experiment_name = f"{cfg['simulator']['name']}_{cfg['planner']['type']}"
    logger = Logger(experiment_name, cfg, model_cfg)

    simulator = get_simulator(cfg)
    simulator.start_mission(np.array([0, 0, 1], dtype=np.float32))

    for _ in range(num_data_samples):
        random_annotation_mask = sample_annotation_mask(simulator.sensor.resolution, num_pixels_per_sample)
        logger.save_anno_mask_to_disk(
            random_annotation_mask, model_cfg["data"]["path_to_dataset"], dataset_folder=dataset_folder
        )


if __name__ == "__main__":
    main()
