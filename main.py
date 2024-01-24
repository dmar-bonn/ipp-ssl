import os
from typing import Dict, Tuple

import click
import numpy as np
import yaml

from active_learning import get_learner
from active_learning.annotators import Annotator
from active_learning.missions import Mission
from mapper import get_mapper
from planner import get_local_navigation_planner, get_planner
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


def get_starting_pose(simulator_name: str, altitude: float, starting_position: str) -> np.array:
    if simulator_name == "potsdam":
        if starting_position == "top_left":
            x_pos, y_pos = 30, 30
        else:
            raise ValueError(f"Starting position '{starting_position}' not found!")
    else:
        raise ValueError(f"Simulator '{simulator_name}' not found!")

    return np.array([x_pos, y_pos, altitude], dtype=np.float32)


@click.command()
@click.option(
    "--config_file",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml"),
)
def main(config_file: str):
    cfg, model_cfg = read_config_files(config_file)

    experiment_name = f"{cfg['simulator']['name']}_{cfg['planner']['type']}"
    logger = Logger(experiment_name, cfg, model_cfg)
    logger.save_config_files_to_disk(cfg, model_cfg)

    simulator = get_simulator(cfg)
    learner = get_learner(model_cfg, cfg["network"]["path_to_checkpoint"], logger.logger_name, model_id="0")
    trained_model = learner.setup_model(checkpoint_path=learner.weights_path)

    for notifier in logger.notifiers:
        notifier.start_experiment()

    try:
        init_pose = get_starting_pose(
            cfg["simulator"]["name"], cfg["planner"]["altitude"], cfg["planner"]["starting_position"]
        )

        for mission_id in range(cfg["planner"]["num_missions"]):
            mapper = get_mapper(cfg, model_cfg, simulator.sensor)
            planner = get_planner(cfg, mapper, mission_id=mission_id)
            local_navigation_planner = get_local_navigation_planner(cfg, mapper)
            mission = Mission(
                planner, local_navigation_planner, mapper, simulator, trained_model, init_pose, cfg, model_cfg, logger
            )
            annotator = Annotator(mapper, simulator, cfg, model_cfg, logger, mission)

            simulator.start_mission(init_pose)
            mission.execute(mission_id)

            logger.remove_train_data_from_disk(
                model_cfg["data"]["path_to_dataset"], dataset_folder=os.path.join("training_set", "pseudo")
            )

            if cfg["annotations"]["human"]["use_human_labels"]:
                annotator.create_human_mission_annotations(mission_id)

            pretrained_model_checkpoint = learner.weights_path
            if cfg["annotations"]["pseudo"]["fine_tuning"]:
                _, pretrained_model_checkpoint = learner.train(mission_id, pretrained_model_checkpoint, True, False)
                if cfg["annotations"]["pseudo"]["map_replay"]:
                    print(f"START MAP EXPERIENCE REPLAY")
                    mission.compute_informed_map_prior(mission_id, store_prior_map=False)

            if cfg["annotations"]["pseudo"]["use_pseudo_labels"]:
                annotator.create_pseudo_mission_annotations(mission_id)

            trained_model, _ = learner.train(
                mission_id,
                pretrained_model_checkpoint,
                cfg["annotations"]["human"]["use_human_labels"],
                cfg["annotations"]["pseudo"]["use_pseudo_labels"],
            )

            learner.evaluate(
                mission_id,
                cfg["annotations"]["human"]["use_human_labels"],
                cfg["annotations"]["pseudo"]["use_pseudo_labels"],
            )

            logger.finished_mission(mission_id, learner.test_statistics)

        for notifier in logger.notifiers:
            notifier.finish_experiment(additional_info=learner.test_statistics)

    except Exception as e:
        for notifier in logger.notifiers:
            notifier.failed_experiment(e)

        raise Exception(e)


if __name__ == "__main__":
    main()
