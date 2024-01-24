import os.path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from agri_semantics.datasets import get_data_module
from agri_semantics.models.models import IGNORE_INDEX
from agri_semantics.utils.utils import toOneHot
from torch import nn

from active_learning.missions import Mission
from mapper.common import Mapper
from simulator import Simulator
from utils.logger import Logger


class Annotator:
    def __init__(
        self,
        mapper: Mapper,
        simulator: Simulator,
        cfg: Dict,
        model_cfg: Dict,
        logger: Logger,
        mission: Mission,
    ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.mapper = mapper
        self.simulator = simulator
        self.logger = logger
        self.mission = mission
        self.path_to_dataset = model_cfg["data"]["path_to_dataset"]

        self.simulator_name = cfg["simulator"]["name"]
        self.sensor_resolution = cfg["simulator"][self.simulator_name]["sensor"]["resolution"]
        self.sensor_angle = cfg["simulator"][self.simulator_name]["sensor"]["angle"]

        self.human_annotations = cfg["annotations"]["human"]
        self.pseudo_annotations = cfg["annotations"]["pseudo"]

        self.num_of_pseudo_pixels_per_class = None
        self.uncertainty_threshold_per_class = None

    @staticmethod
    def get_ground_truth_annotation(measurement: Dict) -> np.array:
        return measurement["anno"]

    def get_prediction_uncertainty(self, measurement: Dict) -> Tuple[np.array, np.array]:
        prediction = self.mission.infer_map_data(measurement)
        return prediction["logits"], prediction["uncertainty"]

    @staticmethod
    def get_maximum_likelihood_annotation(probabilistic_annotation: np.array) -> np.array:
        return np.argmax(probabilistic_annotation, axis=0)

    def mask_uncertain_predictions(self, prediction_probs: np.array, prediction_uncertainties: np.array) -> np.array:
        uncertainty_mask = prediction_uncertainties >= self.pseudo_annotations["uncertainty_threshold"]
        prediction_probs[:, uncertainty_mask] = 0.0
        prediction_probs[IGNORE_INDEX[self.simulator_name], uncertainty_mask] = 1.0
        return prediction_probs

    def sample_random_annotation_mask(self, num_rand_pixels: int, sampling_mask: np.array) -> np.array:
        sampling_indices = np.argwhere(sampling_mask)
        sampled_indices_idx = np.random.choice(len(sampling_indices), size=num_rand_pixels, replace=False)
        sampled_indices = sampling_indices[sampled_indices_idx, :]

        anno_mask = np.zeros((self.sensor_resolution[1], self.sensor_resolution[0]), dtype=np.uint8)
        anno_mask[sampled_indices[:, 0], sampled_indices[:, 1]] = 1
        return anno_mask.astype(bool)

    def sample_uncertainty_aware_annotation_mask_v1(
        self, uncertainties: np.array, pseudo_labels: bool, mission_id: int, sampling_mask: np.array
    ) -> np.array:
        anno_specs = self.pseudo_annotations if pseudo_labels else self.human_annotations
        sorted_ids_x, sorted_ids_y = np.unravel_index(np.argsort(uncertainties.flatten()), uncertainties.shape)
        num_pixels_above_threshold_x = int(anno_specs["sampling_method"]["pixel_share"] * len(sorted_ids_x))
        num_pixels_above_threshold_y = int(anno_specs["sampling_method"]["pixel_share"] * len(sorted_ids_y))
        uncertain_ids_x, uncertain_ids_y = (
            sorted_ids_x[-num_pixels_above_threshold_x:],
            sorted_ids_y[-num_pixels_above_threshold_y:],
        )
        if pseudo_labels:
            uncertain_ids_x, uncertain_ids_y = (
                sorted_ids_x[:num_pixels_above_threshold_x],
                sorted_ids_y[:num_pixels_above_threshold_y],
            )
        uncertain_ids = np.array((uncertain_ids_x, uncertain_ids_y)).T

        sampling_indices = np.argwhere(sampling_mask)
        uncertain_ids_ids, _ = np.where(np.sum(sampling_indices == uncertain_ids[:, None], axis=2) == 2)
        uncertain_ids = uncertain_ids[uncertain_ids_ids, :]

        sampled_uncertain_ids_ids = np.random.choice(
            len(uncertain_ids), size=self.get_pixels_per_image(pseudo_labels, mission_id), replace=False
        )
        sampled_uncertain_ids_x = uncertain_ids[sampled_uncertain_ids_ids, 0]
        sampled_uncertain_ids_y = uncertain_ids[sampled_uncertain_ids_ids, 1]
        anno_mask = np.zeros_like(uncertainties, dtype=np.uint8)
        anno_mask[sampled_uncertain_ids_x, sampled_uncertain_ids_y] = 1

        return anno_mask.astype(bool)

    def sample_region_uncertainty_aware_annotation_mask(
        self, uncertainties: np.array, pseudo_labels: bool, mission_id: int, sampling_mask: np.array
    ) -> np.array:
        anno_specs = self.pseudo_annotations if pseudo_labels else self.human_annotations

        conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=anno_specs["sampling_method"]["kernel_size"],
            stride=(1, 1),
            padding=int(anno_specs["sampling_method"]["kernel_size"] / 2),
            bias=False,
            padding_mode="zeros",
            groups=1,
        )
        weight = torch.ones(
            (anno_specs["sampling_method"]["kernel_size"], anno_specs["sampling_method"]["kernel_size"]),
            dtype=torch.float32,
        )
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
        conv.weight = nn.Parameter(weight)
        conv.requires_grad_(False)

        uncertainties_per_region = (
            conv(torch.from_numpy(uncertainties).unsqueeze(dim=0)).squeeze(dim=0).cpu().detach().numpy()
        )

        sorted_ids_x, sorted_ids_y = np.unravel_index(
            np.argsort(uncertainties_per_region.flatten()), uncertainties_per_region.shape
        )
        num_pixels_above_threshold_x = int(anno_specs["sampling_method"]["pixel_share"] * len(sorted_ids_x))
        num_pixels_above_threshold_y = int(anno_specs["sampling_method"]["pixel_share"] * len(sorted_ids_y))
        uncertain_ids_x, uncertain_ids_y = (
            sorted_ids_x[-num_pixels_above_threshold_x:],
            sorted_ids_y[-num_pixels_above_threshold_y:],
        )
        if pseudo_labels:
            uncertain_ids_x, uncertain_ids_y = (
                sorted_ids_x[:num_pixels_above_threshold_x],
                sorted_ids_y[:num_pixels_above_threshold_y],
            )
        uncertain_ids = np.array((uncertain_ids_x, uncertain_ids_y)).T

        sampling_indices = np.argwhere(sampling_mask)
        uncertain_ids_ids, _ = np.where(np.sum(sampling_indices == uncertain_ids[:, None], axis=2) == 2)
        uncertain_ids = uncertain_ids[uncertain_ids_ids, :]

        sampled_uncertain_ids_ids = np.random.choice(
            len(uncertain_ids), size=self.get_pixels_per_image(pseudo_labels, mission_id), replace=False
        )
        sampled_uncertain_ids_x = uncertain_ids[sampled_uncertain_ids_ids, 0]
        sampled_uncertain_ids_y = uncertain_ids[sampled_uncertain_ids_ids, 1]
        anno_mask = np.zeros_like(uncertainties_per_region, dtype=np.uint8)
        anno_mask[sampled_uncertain_ids_x, sampled_uncertain_ids_y] = 1

        return anno_mask.astype(bool)

    def sample_uncertainty_aware_annotation_mask_v2(
        self, uncertainties: np.array, pseudo_labels: bool, mission_id: int, sampling_mask: np.array
    ) -> np.array:
        anno_specs = self.pseudo_annotations if pseudo_labels else self.human_annotations
        pixels_per_image = self.get_pixels_per_image(pseudo_labels, mission_id)
        num_rand_pixels = int(anno_specs["sampling_method"]["pixel_share"] * np.prod(uncertainties.shape))

        random_anno_mask = self.sample_random_annotation_mask(num_rand_pixels, sampling_mask)
        sampled_random_ids = np.argwhere(random_anno_mask == 1)

        sorted_random_ids = np.argsort(uncertainties[sampled_random_ids[:, 0], sampled_random_ids[:, 1]])
        uncertain_random_ids = sampled_random_ids[sorted_random_ids[-pixels_per_image:]]
        if pseudo_labels:
            uncertain_random_ids = sampled_random_ids[sorted_random_ids[:pixels_per_image]]

        anno_mask = np.zeros_like(uncertainties, dtype=np.uint8)
        anno_mask[uncertain_random_ids[:, 0], uncertain_random_ids[:, 1]] = 1

        return anno_mask.astype(bool)

    def create_distribution_aligned_annotation_mask(
        self, prediction_probs: np.array, uncertainties: np.array, mission_id: int, sampling_mask: np.array
    ) -> np.array:
        if self.num_of_pseudo_pixels_per_class is None or self.uncertainty_threshold_per_class is None:
            (
                self.num_of_pseudo_pixels_per_class,
                self.uncertainty_threshold_per_class,
            ) = self.compute_pseudo_label_uncertainty_thresholds(self.logger.all_train_poses, mission_id)

        mle_anno = self.get_maximum_likelihood_annotation(prediction_probs)
        anno_mask = np.zeros((self.sensor_resolution[1], self.sensor_resolution[0]), dtype=np.uint8)

        for class_id, uncertainty_threshold in self.uncertainty_threshold_per_class.items():
            uncertainty_msk = uncertainties <= uncertainty_threshold
            class_msk = mle_anno == class_id

            if np.sum(class_msk & uncertainty_msk) == 0:
                continue

            anno_mask[uncertainty_msk & class_msk & sampling_mask] = 1

        return anno_mask.astype(bool)

    def compute_region_impurity(self, prediction_probs: np.array, anno_specs: Dict) -> np.array:
        mle_prediction = self.get_maximum_likelihood_annotation(prediction_probs)
        mle_prediction = torch.from_numpy(mle_prediction)
        prediction_one_hot = (
            nn.functional.one_hot(mle_prediction, num_classes=self.mapper.class_num).movedim(-1, 0).float()
        )

        conv = nn.Conv2d(
            in_channels=self.mapper.class_num,
            out_channels=self.mapper.class_num,
            kernel_size=anno_specs["sampling_method"]["kernel_size"],
            stride=(1, 1),
            padding=int(anno_specs["sampling_method"]["kernel_size"] / 2),
            bias=False,
            padding_mode="zeros",
            groups=self.mapper.class_num,
        )
        weight = torch.ones(
            (anno_specs["sampling_method"]["kernel_size"], anno_specs["sampling_method"]["kernel_size"]),
            dtype=torch.float32,
        )
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
        weight = weight.repeat([self.mapper.class_num, 1, 1, 1])
        conv.weight = nn.Parameter(weight)
        conv.requires_grad_(False)

        per_class_count_per_region = conv(prediction_one_hot)
        class_region_dist = per_class_count_per_region / torch.sum(per_class_count_per_region, dim=0, keepdim=True)
        region_impurity = torch.sum(-class_region_dist * torch.log(class_region_dist + 1e-6), dim=0)
        return region_impurity.cpu().detach().numpy()

    def create_region_impurity_annotation_mask(
        self, prediction_probs: np.array, pseudo_labels: bool, mission_id: int, sampling_mask: np.array
    ) -> np.array:
        anno_specs = self.pseudo_annotations if pseudo_labels else self.human_annotations
        pixels_per_image = self.get_pixels_per_image(pseudo_labels, mission_id)
        region_impurity = self.compute_region_impurity(prediction_probs, anno_specs)
        region_impurity[~sampling_mask] = -np.inf
        sorted_ids_x, sorted_ids_y = np.unravel_index(np.argsort(region_impurity.flatten()), region_impurity.shape)

        anno_mask = np.zeros_like(region_impurity, dtype=np.uint8)
        anno_mask[sorted_ids_x[-pixels_per_image:], sorted_ids_y[-pixels_per_image:]] = 1
        return anno_mask.astype(bool)

    def sample_region_impurity_annotation_mask(
        self, prediction_probs: np.array, pseudo_labels: bool, mission_id: int, sampling_mask: np.array
    ) -> np.array:
        anno_specs = self.pseudo_annotations if pseudo_labels else self.human_annotations

        region_impurity = self.compute_region_impurity(prediction_probs, anno_specs)
        sorted_ids_x, sorted_ids_y = np.unravel_index(np.argsort(region_impurity.flatten()), region_impurity.shape)

        num_pixels_above_threshold_x = int(anno_specs["sampling_method"]["pixel_share"] * len(sorted_ids_x))
        num_pixels_above_threshold_y = int(anno_specs["sampling_method"]["pixel_share"] * len(sorted_ids_y))
        reg_imp_ids_x, reg_imp_ids_y = (
            sorted_ids_x[-num_pixels_above_threshold_x:],
            sorted_ids_y[-num_pixels_above_threshold_y:],
        )
        if pseudo_labels:
            reg_imp_ids_x, reg_imp_ids_y = (
                sorted_ids_x[:num_pixels_above_threshold_x],
                sorted_ids_y[:num_pixels_above_threshold_y],
            )
        reg_imp_ids = np.array((reg_imp_ids_x, reg_imp_ids_y)).T

        sampling_indices = np.argwhere(sampling_mask)
        reg_imp_ids_ids, _ = np.where(np.sum(sampling_indices == reg_imp_ids[:, None], axis=2) == 2)
        reg_imp_ids = reg_imp_ids[reg_imp_ids_ids, :]

        sampled_reg_imp_ids_ids = np.random.choice(
            len(reg_imp_ids), size=self.get_pixels_per_image(pseudo_labels, mission_id), replace=False
        )
        sampled_reg_imp_ids_x = reg_imp_ids[sampled_reg_imp_ids_ids, 0]
        sampled_reg_imp_ids_y = reg_imp_ids[sampled_reg_imp_ids_ids, 1]

        anno_mask = np.zeros_like(region_impurity, dtype=np.uint8)
        anno_mask[sampled_reg_imp_ids_x, sampled_reg_imp_ids_y] = 1
        return anno_mask.astype(bool)

    def get_pixels_per_image(self, pseudo_labels: bool, mission_id: int):
        anno_specs = self.pseudo_annotations if pseudo_labels else self.human_annotations
        if not pseudo_labels:
            return anno_specs["pixels_per_image"]

        return int(
            np.linspace(
                anno_specs["min_pixels_per_image"],
                anno_specs["max_pixels_per_image"],
                self.cfg["planner"]["num_missions"],
            )[mission_id]
        )

    def get_annotation_mask(
        self,
        prediction_probs: np.array,
        uncertainties: np.array,
        pseudo_labels: bool,
        mission_id: int,
        sampling_mask: np.array,
    ) -> np.array:
        anno_specs = self.pseudo_annotations if pseudo_labels else self.human_annotations
        if not anno_specs["sparse_annotations"]:
            return np.ones((self.sensor_resolution[1], self.sensor_resolution[0])).astype(bool)

        if anno_specs["sampling_method"]["name"] == "random":
            return self.sample_random_annotation_mask(
                self.get_pixels_per_image(pseudo_labels, mission_id), sampling_mask
            )
        elif anno_specs["sampling_method"]["name"] == "uncertainty_random":
            return self.sample_uncertainty_aware_annotation_mask_v1(
                uncertainties, pseudo_labels, mission_id, sampling_mask
            )
        elif anno_specs["sampling_method"]["name"] == "region_uncertainty_random":
            return self.sample_region_uncertainty_aware_annotation_mask(
                uncertainties, pseudo_labels, mission_id, sampling_mask
            )
        elif anno_specs["sampling_method"]["name"] == "random_uncertainty":
            return self.sample_uncertainty_aware_annotation_mask_v2(
                uncertainties, pseudo_labels, mission_id, sampling_mask
            )
        elif anno_specs["sampling_method"]["name"] == "distribution_alignment":
            return self.create_distribution_aligned_annotation_mask(
                prediction_probs, uncertainties, mission_id, sampling_mask
            )
        elif anno_specs["sampling_method"]["name"] == "region_impurity":
            return self.create_region_impurity_annotation_mask(
                prediction_probs, pseudo_labels, mission_id, sampling_mask
            )
        elif anno_specs["sampling_method"]["name"] == "region_impurity_random":
            return self.sample_region_impurity_annotation_mask(
                prediction_probs, pseudo_labels, mission_id, sampling_mask
            )
        else:
            raise ValueError(
                f"Sparse annotation sampling method '{anno_specs['sampling_method']['name']}' does not exist!"
            )

    def compute_annotation_and_uncertainty(
        self,
        measurement: Dict,
        mission_id: int,
        pseudo_labels: bool,
    ) -> Tuple[np.array, np.array, np.array]:
        anno_specs = self.pseudo_annotations if pseudo_labels else self.human_annotations

        if anno_specs["uncertainty_source"] == "map":
            prediction_probs, uncertainties, _, _, _ = self.mapper.get_map_state(
                measurement["pose"], depth=measurement["depth"]
            )
        else:
            prediction_probs, uncertainties = self.get_prediction_uncertainty(measurement)

        if pseudo_labels:
            prediction_probs = self.mask_uncertain_predictions(prediction_probs, uncertainties)
        mle_anno = self.get_maximum_likelihood_annotation(prediction_probs)

        sampling_mask = np.ones((self.sensor_resolution[1], self.sensor_resolution[0]), dtype=bool)
        if pseudo_labels:
            sampling_mask = mle_anno != IGNORE_INDEX[self.simulator_name]

        anno_rgb_img = toOneHot(torch.from_numpy(mle_anno).unsqueeze(0), self.mapper.map_name)
        anno_rgb_img = cv2.cvtColor(anno_rgb_img, cv2.COLOR_RGB2BGR)

        anno_mask = self.get_annotation_mask(prediction_probs, uncertainties, pseudo_labels, mission_id, sampling_mask)
        if not pseudo_labels:
            anno_rbg_img = self.get_ground_truth_annotation(measurement)
            uncertainty_img = np.zeros((self.sensor_resolution[1], self.sensor_resolution[0]), dtype=np.uint8)
            return anno_rbg_img, anno_mask, uncertainty_img

        uncertainty_img = (uncertainties * 255).astype(np.uint8)

        return anno_rgb_img, anno_mask, uncertainty_img

    def create_annotations_from_measurements(self, measurements: List, mission_id: int, pseudo_labels: bool):
        dataset_subfolder = "pseudo" if pseudo_labels else "human"
        dataset_folder = os.path.join("training_set", dataset_subfolder)

        for measurement_id, measurement in enumerate(measurements):
            anno, anno_mask, uncertainty = self.compute_annotation_and_uncertainty(
                measurement, mission_id, pseudo_labels
            )
            self.logger.save_anno_to_disk(anno, self.path_to_dataset, dataset_folder=dataset_folder)
            self.logger.save_anno_mask_to_disk(anno_mask, self.path_to_dataset, dataset_folder=dataset_folder)
            self.logger.save_uncertainty_to_disk(uncertainty, self.path_to_dataset, dataset_folder=dataset_folder)

            pseudo_anno = anno if pseudo_labels else None
            self.logger.save_qualitative_anno_mask(anno_mask, mission_id, measurement_id, pseudo_anno=pseudo_anno)

    def compute_human_class_frequencies(self) -> Dict:
        dataloader = get_data_module(self.model_cfg)
        dataloader.setup(stage=None)
        human_class_counts = self.logger.compute_class_counts(dataloader.val_dataloader())

        if 0 <= IGNORE_INDEX[self.simulator_name] < len(human_class_counts):
            human_class_counts[IGNORE_INDEX[self.simulator_name]] = 0.0

        total_num_pixels = torch.sum(human_class_counts).item()
        return {
            class_id: class_count.item() / total_num_pixels for class_id, class_count in enumerate(human_class_counts)
        }

    def compute_pseudo_label_uncertainty_thresholds(
        self, pseudo_measurements: List, mission_id: int
    ) -> Tuple[Dict, Dict]:
        alpha = self.get_pixels_per_image(True, mission_id) / np.prod(self.sensor_resolution)
        total_pseudo_pixels = len(pseudo_measurements) * np.prod(self.sensor_resolution)
        human_class_frequencies = self.compute_human_class_frequencies()
        num_of_pseudo_pixels_per_class = {}
        for class_id, class_freq in human_class_frequencies.items():
            num_of_pseudo_pixels_per_class[class_id] = int(alpha * total_pseudo_pixels * class_freq)

        all_prediction_probs = np.zeros(
            (len(pseudo_measurements), self.mapper.class_num, self.sensor_resolution[1], self.sensor_resolution[0])
        )
        all_uncertainties = np.zeros((len(pseudo_measurements), self.sensor_resolution[1], self.sensor_resolution[0]))
        for measurement_id, measurement in enumerate(pseudo_measurements):
            anno_map_probs, uncertainties_map, _, _, _ = self.mapper.get_map_state(
                measurement["pose"], depth=measurement["depth"]
            )
            uncertainties = uncertainties_map
            if self.pseudo_annotations["uncertainty_source"] == "prediction":
                _, uncertainties = self.get_prediction_uncertainty(measurement)

            all_prediction_probs[measurement_id, :, :] = anno_map_probs
            all_uncertainties[measurement_id, :, :] = uncertainties

        all_predictions = np.argmax(all_prediction_probs, axis=1)

        sorted_unc_ids_x, sorted_unc_ids_y, sorted_unc_ids_z = np.unravel_index(
            np.argsort(all_uncertainties.flatten()), all_uncertainties.shape
        )
        sorted_all_predictions = all_predictions[sorted_unc_ids_x, sorted_unc_ids_y, sorted_unc_ids_z]
        sorted_all_uncertainties = all_uncertainties[sorted_unc_ids_x, sorted_unc_ids_y, sorted_unc_ids_z]

        uncertainty_threshold_per_class = {}
        for class_id, num_pixels in num_of_pseudo_pixels_per_class.items():
            class_msk = sorted_all_predictions == class_id

            if num_of_pseudo_pixels_per_class[class_id] <= 0:
                uncertainty_threshold_per_class[class_id] = -np.inf
                continue

            if num_of_pseudo_pixels_per_class[class_id] > np.sum(class_msk) or np.sum(class_msk) == 0:
                uncertainty_threshold_per_class[class_id] = np.inf
                continue

            uncertainty_threshold_per_class[class_id] = sorted_all_uncertainties[class_msk][
                num_of_pseudo_pixels_per_class[class_id] - 1
            ]

        return num_of_pseudo_pixels_per_class, uncertainty_threshold_per_class

    def get_pseudo_label_measurements(self) -> List:
        if self.pseudo_annotations["image_source"] == "waypoint":
            return self.logger.all_train_poses
        elif self.pseudo_annotations["image_source"] == "in_between":
            return self.logger.all_waypoints
        else:
            raise ValueError(f"Pseudo label image source '{self.pseudo_annotations['image_source']}' does not exist!")

    def create_human_mission_annotations(self, mission_id: int):
        print("CREATE HUMAN ANNOTATIONS")
        self.create_annotations_from_measurements(self.logger.mission_train_poses, mission_id, pseudo_labels=False)
        file_id = f"{self.cfg['simulator']['name']}_{self.cfg['planner']['type']}_{mission_id}"
        self.logger.save_train_data_stats(file_id)

    def create_pseudo_mission_annotations(self, mission_id: int):
        print("CREATE PSEUDO ANNOTATIONS")
        self.create_annotations_from_measurements(self.get_pseudo_label_measurements(), mission_id, pseudo_labels=True)
