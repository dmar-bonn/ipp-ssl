import os
from typing import Dict, List

import cv2
import matplotlib
import numpy as np
import seaborn as sns
import torch
import yaml
from agri_semantics.datasets import get_data_module
from agri_semantics.models.models import IGNORE_INDEX
from agri_semantics.utils.utils import toOneHot
from torch.utils.data import DataLoader

from utils.notifiers import Notifier
from utils.notifiers.slack import SlackNotifier
from utils.notifiers.telegram import TelegramNotifier
from utils.utils import load_from_env

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, experiment_name: str, cfg: Dict, model_cfg: Dict):
        self.logger_name = experiment_name

        self.setup_log_dir()

        self.all_waypoints = []

        self.mission_train_poses = []
        self.all_train_poses = []

        self.cfg = cfg
        self.model_cfg = model_cfg

        self.verbose = cfg["logging"]["verbose"]
        self.simulator_name = cfg["simulator"]["name"]

        self.notifiers = self.setup_notifiers()

    def setup_notifiers(self) -> List[Notifier]:
        notifiers = []
        if self.cfg["notifications"]["telegram"]["is_used"]:
            notifiers.append(
                TelegramNotifier(
                    self.logger_name,
                    load_from_env("TELEGRAM_TOKEN", str, "my_telegram_token"),
                    load_from_env("TELEGRAM_CHAT_ID", str, "my_telegram_chat_id"),
                    cfg=self.cfg,
                    model_cfg=self.model_cfg,
                    verbose=self.cfg["notifications"]["telegram"]["verbose"],
                )
            )

        if self.cfg["notifications"]["slack"]["is_used"]:
            notifiers.append(
                SlackNotifier(
                    self.logger_name,
                    load_from_env("SLACK_WEBHOOK", str, "my_slack_webhook"),
                    load_from_env("SLACK_BOTNAME", str, "my_slack_botname"),
                    icon=self.cfg["notifications"]["slack"]["icon"],
                    cfg=self.cfg,
                    model_cfg=self.model_cfg,
                    verbose=self.cfg["notifications"]["slack"]["verbose"],
                )
            )

        return notifiers

    def finished_mission(self, mission_id: int, test_statistics: Dict):
        self.save_evaluation_metrics_to_disk(test_statistics)
        self.reset_mission_train_poses()

        for notifier in self.notifiers:
            notifier.finished_iteration(mission_id, additional_info=test_statistics)

    def setup_log_dir(self):
        if os.path.exists(self.logger_name):
            raise ValueError(f"{self.logger_name} log directory already exists!")

        os.makedirs(self.logger_name)

    def reset_mission_train_poses(self):
        self.mission_train_poses = []

    def add_waypoint(self, waypoint: np.array, measurement: Dict):
        self.all_waypoints.append(measurement)

    def add_train_pose(self, measurement: Dict):
        self.mission_train_poses.append(measurement)
        self.all_train_poses.append(measurement)

    @staticmethod
    def save_anno_to_disk(
        anno: np.array, dataset_path: str, dataset_folder: str = os.path.join("training_set", "human")
    ):
        anno_dir = os.path.join(dataset_path, dataset_folder, "anno")
        if not os.path.exists(anno_dir):
            os.makedirs(anno_dir)

        train_data_id = len([name for name in os.listdir(anno_dir) if os.path.isfile(os.path.join(anno_dir, name))])
        anno_filepath = os.path.join(anno_dir, f"gt_{str(train_data_id).zfill(5)}.png")
        cv2.imwrite(anno_filepath, anno)

    @staticmethod
    def save_anno_mask_to_disk(
        anno_mask: np.array, dataset_path: str, dataset_folder: str = os.path.join("training_set", "human")
    ):
        mask_dir = os.path.join(dataset_path, dataset_folder, "anno_mask")
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        train_data_id = len([name for name in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, name))])
        mask_filepath = os.path.join(mask_dir, f"gt_mask_{str(train_data_id).zfill(5)}.npy")
        np.save(mask_filepath, anno_mask)

    @staticmethod
    def save_uncertainty_to_disk(
        uncertainty: np.array, dataset_path: str, dataset_folder: str = os.path.join("training_set", "human")
    ):
        unc_dir = os.path.join(dataset_path, dataset_folder, "uncertainty")
        if not os.path.exists(unc_dir):
            os.makedirs(unc_dir)

        train_data_id = len([name for name in os.listdir(unc_dir) if os.path.isfile(os.path.join(unc_dir, name))])
        unc_filepath = os.path.join(unc_dir, f"unc_{str(train_data_id).zfill(5)}.png")
        cv2.imwrite(unc_filepath, uncertainty)

    @staticmethod
    def save_image_to_disk(
        image: np.array, dataset_path: str, dataset_folder: str = os.path.join("training_set", "human")
    ):
        image_dir = os.path.join(dataset_path, dataset_folder, "image")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        train_data_id = len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])
        image_filepath = os.path.join(image_dir, f"rgb_{str(train_data_id).zfill(5)}.png")

        if image.shape[2] > 4:
            np.save(image_filepath, image)
        else:
            cv2.imwrite(image_filepath, image)

    def save_train_data_to_disk(
        self,
        image: np.array,
        anno: np.array,
        dataset_path: str,
        dataset_folder: str = os.path.join("training_set", "human"),
        uncertainty: np.array = None,
    ):
        self.save_image_to_disk(image, dataset_path, dataset_folder)
        self.save_anno_to_disk(anno, dataset_path, dataset_folder)

        if uncertainty is not None:
            self.save_uncertainty_to_disk(uncertainty, dataset_path, dataset_folder)

    @staticmethod
    def remove_annos_from_disk(dataset_path: str, dataset_folder: str = os.path.join("training_set", "human")):
        anno_dir = os.path.join(dataset_path, dataset_folder, "anno")
        if not os.path.exists(anno_dir):
            return

        for filename in os.listdir(anno_dir):
            os.remove(os.path.join(anno_dir, filename))

    @staticmethod
    def remove_anno_masks_from_disk(dataset_path: str, dataset_folder: str = os.path.join("training_set", "human")):
        anno_dir = os.path.join(dataset_path, dataset_folder, "anno_mask")
        if not os.path.exists(anno_dir):
            return

        for filename in os.listdir(anno_dir):
            os.remove(os.path.join(anno_dir, filename))

    @staticmethod
    def remove_uncertainties_from_disk(dataset_path: str, dataset_folder: str = os.path.join("training_set", "human")):
        unc_dir = os.path.join(dataset_path, dataset_folder, "uncertainty")
        if not os.path.exists(unc_dir):
            return

        for filename in os.listdir(unc_dir):
            os.remove(os.path.join(unc_dir, filename))

    @staticmethod
    def remove_images_from_disk(dataset_path: str, dataset_folder: str = os.path.join("training_set", "human")):
        image_dir = os.path.join(dataset_path, dataset_folder, "image")
        if not os.path.exists(image_dir):
            return

        for filename in os.listdir(image_dir):
            os.remove(os.path.join(image_dir, filename))

    def remove_train_data_from_disk(
        self, dataset_path: str, dataset_folder: str = os.path.join("training_set", "human")
    ):
        self.remove_annos_from_disk(dataset_path, dataset_folder)
        self.remove_anno_masks_from_disk(dataset_path, dataset_folder)
        self.remove_uncertainties_from_disk(dataset_path, dataset_folder)

    def save_qualitative_anno_mask(
        self,
        anno_mask: np.array,
        mission_id: int,
        mask_id: int,
        pseudo_anno: np.array = None,
    ):
        if not os.path.exists(os.path.join(self.logger_name, "qualitative_results")):
            os.makedirs(os.path.join(self.logger_name, "qualitative_results"))

        sub_folder = "pseudo" if pseudo_anno is not None else "human"
        folder_path = os.path.join(self.logger_name, "qualitative_results", sub_folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        folder_path = os.path.join(self.logger_name, "qualitative_results", sub_folder, f"mission_{mission_id}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, f"anno_mask_{mask_id}.npy"), anno_mask)
        if pseudo_anno is not None:
            cv2.imwrite(os.path.join(folder_path, f"pseudo_anno_{mask_id}.png"), pseudo_anno)

    def save_qualitative_results(
        self,
        image: np.array,
        anno: np.array,
        map_data: Dict,
        mission_id: int,
        timestep: int,
        map_name: str,
        pseudo_labels: bool = False,
    ):
        if not os.path.exists(os.path.join(self.logger_name, "qualitative_results")):
            os.makedirs(os.path.join(self.logger_name, "qualitative_results"))

        sub_folder = "pseudo" if pseudo_labels else "human"
        folder_path = os.path.join(self.logger_name, "qualitative_results", sub_folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        folder_path = os.path.join(self.logger_name, "qualitative_results", sub_folder, f"mission_{mission_id}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        cv2.imwrite(os.path.join(folder_path, f"image_{timestep}.png"), image[:, :, :3])
        cv2.imwrite(os.path.join(folder_path, f"anno_{timestep}.png"), anno)

        if self.model_cfg["model"]["task"] == "classification":
            prediction = toOneHot(torch.from_numpy(map_data["logits"]).unsqueeze(0), map_name)
            cv2.imwrite(os.path.join(folder_path, f"pred_{timestep}.png"), cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(os.path.join(folder_path, f"pred_{timestep}.png"), map_data["logits"])

        plt.imsave(
            os.path.join(folder_path, f"uncertainty_{timestep}.png"), np.squeeze(map_data["uncertainty"]), cmap="plasma"
        )

    def save_maps_to_disk(
        self,
        semantic_map: np.array,
        epistemic_map: np.array,
        file_id: str,
        map_name: str,
    ):
        if map_name not in ["flightmare", "habitat"]:
            if semantic_map.shape[0] > 1:
                semantic_map_name = "semantics"
                plt.imsave(
                    os.path.join(self.logger_name, f"{semantic_map_name}_{file_id}.png"),
                    toOneHot(torch.from_numpy(semantic_map).unsqueeze(0), map_name),
                )
            else:
                semantic_map_name = "elevation"
                plt.imsave(
                    os.path.join(self.logger_name, f"{semantic_map_name}_{file_id}.png"),
                    np.squeeze(semantic_map),
                    cmap="gray",
                )

            plt.imsave(
                os.path.join(self.logger_name, f"uncertainty_{file_id}.png"), np.squeeze(epistemic_map), cmap="plasma"
            )

        if self.verbose:
            with open(os.path.join(self.logger_name, f"semantic_map_{file_id}.npy"), "wb") as file:
                np.save(file, semantic_map)

            with open(os.path.join(self.logger_name, f"epistemic_map_{file_id}.npy"), "wb") as file:
                np.save(file, epistemic_map)

        plt.clf()
        plt.cla()

    def save_path_to_disk(self, file_id: str):
        mission_train_poses = np.array([measurement["pose"] for measurement in self.mission_train_poses])
        plt.plot(mission_train_poses[:, 0], mission_train_poses[:, 1], "-ok")
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(self.logger_name, f"path_{file_id}.png"))

        if self.verbose:
            with open(os.path.join(self.logger_name, f"path_poses_{file_id}.npy"), "wb") as file:
                np.save(file, np.array(mission_train_poses))

        plt.clf()
        plt.cla()

    def save_config_files_to_disk(self, cfg: Dict, model_cfg: Dict):
        with open(os.path.join(self.logger_name, "config.yaml"), "w") as file:
            yaml.dump(cfg, file)

        with open(os.path.join(self.logger_name, "model_config.yaml"), "w") as file:
            yaml.dump(model_cfg, file)

    def save_evaluation_metrics_to_disk(self, test_statistics: Dict):
        with open(os.path.join(self.logger_name, "evaluation_metrics.yaml"), "w") as file:
            yaml.dump(test_statistics, file)

    def compute_class_counts(self, dataloader: DataLoader) -> torch.Tensor:
        all_class_counts = torch.zeros(self.model_cfg["model"]["num_classes"])
        for batch in dataloader:
            class_ids, class_counts = torch.unique(batch["anno"], return_counts=True)
            all_class_counts[class_ids] += class_counts

        return all_class_counts

    def save_train_data_stats(self, file_id: str):
        if self.model_cfg["model"]["task"] == "regression":
            return

        dataloader = get_data_module(self.model_cfg)
        dataloader.setup(stage=None)

        train_data_counts = self.compute_class_counts(dataloader.train_dataloader()["human"])
        if 0 <= IGNORE_INDEX[self.simulator_name] < len(train_data_counts):
            train_data_counts[IGNORE_INDEX[self.simulator_name]] = 0.0

        total_num_pixels = torch.sum(train_data_counts).item()
        train_data_stats = {
            class_id: class_count.item() / total_num_pixels * 100
            for class_id, class_count in enumerate(train_data_counts)
            if class_id != IGNORE_INDEX[self.simulator_name]
        }

        ax = sns.barplot(x=list(train_data_stats.keys()), y=list(train_data_stats.values()))
        ax.set_xlabel("Class Index")
        ax.set_ylabel("Class Frequency [%]")
        plt.savefig(os.path.join(self.logger_name, f"train_data_stats_{file_id}.png"), dpi=300)

        plt.clf()
        plt.cla()
