from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from simulator import Sensor


def get_map_index_tuple(map_indices: np.array) -> Tuple:
    if map_indices.shape[1] == 3:
        return map_indices[:, 0], map_indices[:, 1], map_indices[:, 2]

    return map_indices[:, 0], map_indices[:, 1]


def get_map_boundary_tuple(map_boundary: List) -> Tuple:
    if len(map_boundary) == 3:
        return map_boundary[0], map_boundary[1], map_boundary[2]

    return map_boundary[0], map_boundary[1]


class CountMap:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.map_boundary = cfg["mapper"]["map_boundary"]
        self.count_map = self.init_map()

    def init_map(self) -> np.array:
        return np.zeros(get_map_boundary_tuple(self.map_boundary), dtype=np.int16)

    def update(self, map_indices: np.array):
        self.count_map[get_map_index_tuple(map_indices)] += 1


class DiscreteVariableMap:
    def __init__(self, cfg: Dict, model_cfg: Dict, num_classes: int):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.num_classes = num_classes

        self.map_boundary = cfg["mapper"]["map_boundary"]
        self.log_odds_map, self.prob_map, self.mle_map = self.init_map()
        self.prob_map_outdated, self.mle_map_outdated = False, False

    def init_map(self) -> Tuple[np.array, np.array, np.array]:
        log_odds_prior_map = self.log_odds_prior_const * np.ones(
            (self.num_classes, *get_map_boundary_tuple(self.map_boundary)),
            dtype=np.float16,
        )
        prob_prior_map = (1 / self.num_classes) * np.ones(
            (self.num_classes, *get_map_boundary_tuple(self.map_boundary)),
            dtype=np.float16,
        )
        mle_prior_map = np.zeros(get_map_boundary_tuple(self.map_boundary), dtype=np.float16)
        return log_odds_prior_map, prob_prior_map, mle_prior_map

    @property
    def log_odds_prior_const(self) -> float:
        prob_prior = 1 / self.num_classes
        return np.log(prob_prior / (1 - prob_prior))

    def get_prob_map(self) -> np.array:
        if self.prob_map_outdated:
            self.set_prob_map()
            self.prob_map_outdated = False

        return self.prob_map

    def set_mle_map(self):
        self.mle_map_outdated = False
        self.mle_map = np.argmax(self.log_odds_map, axis=0)

    def set_prob_map(self):
        self.prob_map = 1 - (1 / (1 + np.exp(self.log_odds_map)))

    def update(self, map_indices: np.array, probs_measured: np.array, **kwargs):
        self.mle_map_outdated = True
        self.prob_map_outdated = True

        map_index_tuple_sliced = (slice(None), *get_map_index_tuple(map_indices))
        probs_measured = np.clip(probs_measured, a_min=10 ** (-2), a_max=1 - 10 ** (-2))
        probs_measured /= np.sum(probs_measured, axis=0)
        log_odds_measured = np.log(probs_measured / (1 - probs_measured))
        self.log_odds_map[map_index_tuple_sliced] += log_odds_measured - self.log_odds_prior_const


class ContinuousVariableMap:
    def __init__(
        self,
        cfg: Dict,
        model_cfg: Dict,
        hit_map: CountMap,
        update_type: str,
        num_dimensions: int,
        occupancy_map: bool = False,
    ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.update_type = update_type
        self.num_dimensions = num_dimensions
        self.occupancy_map = occupancy_map

        self.map_boundary = cfg["mapper"]["map_boundary"]
        self.mean_map, self.variance_map, self.hit_map = self.init_map(hit_map)

    def init_map(self, hit_map: CountMap) -> Tuple[np.array, np.array, CountMap]:
        mean_map = self.mean_prior_const * np.ones(
            (self.num_dimensions, *get_map_boundary_tuple(self.map_boundary)),
            dtype=np.float16,
        )
        variance_map = self.variance_prior_const * np.ones(get_map_boundary_tuple(self.map_boundary), dtype=np.float16)
        return mean_map, variance_map, hit_map

    @property
    def mean_prior_const(self) -> float:
        if self.occupancy_map:
            return 0.5

        if self.update_type == "map":
            return (
                self.model_cfg["model"]["value_range"]["min_value"]
                + (
                    self.model_cfg["model"]["value_range"]["max_value"]
                    - self.model_cfg["model"]["value_range"]["min_value"]
                )
                / 2
            )

        return 0

    @property
    def variance_prior_const(self) -> float:
        if self.update_type == "map":
            return (
                self.model_cfg["model"]["value_range"]["max_value"]
                - self.model_cfg["model"]["value_range"]["min_value"]
            ) / 2

        return 0

    @property
    def semantic_map(self) -> np.array:
        return self.mean_map

    @property
    def uncertainty_map(self) -> np.array:
        return self.variance_map

    def update(self, map_indices: np.array, mean_measured: np.array, variance_measured: np.array = None):
        map_index_tuple = get_map_index_tuple(map_indices)
        map_index_tuple_sliced = (slice(None), *map_index_tuple)

        if self.update_type == "mle":
            self.mean_map[map_index_tuple_sliced] = self.maximum_likelihood_update(
                mean_measured,
                self.mean_map[map_index_tuple_sliced],
                self.hit_map.count_map[map_index_tuple],
            )
        elif self.update_type == "map":
            if variance_measured is None:
                raise ValueError(f"Bayesian continuous variable map update requires measurement variance!")
            (
                self.mean_map[map_index_tuple_sliced],
                self.variance_map[map_index_tuple],
            ) = self.kalman_update(
                mean_measured,
                self.mean_map[map_index_tuple_sliced],
                variance_measured,
                self.variance_map[map_index_tuple],
            )
        else:
            raise NotImplementedError(f"Continuous variable map update type '{self.update_type}' not implemented!")

    @staticmethod
    def maximum_likelihood_update(mean_measured: np.array, mean_prior: np.array, hit_map: np.array) -> np.array:
        return mean_prior + (mean_measured - mean_prior) / hit_map

    @staticmethod
    def kalman_update(
        mean_measured: np.array, mean_prior: np.array, variance_measured: np.array, variance_prior: np.array
    ) -> Tuple[np.array, np.array]:
        kalman_gain = variance_prior / (variance_prior + variance_measured + 10 ** (-8))
        mean_post = mean_prior + kalman_gain * (mean_measured - mean_prior)
        variance_post = (1 - kalman_gain) * variance_prior

        return mean_post, variance_post


class Mapper:
    def __init__(self, cfg: Dict, model_cfg: Dict, simulator_name: str, sensor: Sensor, use_torch: bool = False):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.map_name = cfg["mapper"]["map_name"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sensor = sensor
        self.use_torch = use_torch

        self.map_boundary = np.array(cfg["mapper"]["map_boundary"])
        self.world_range = np.array(cfg["simulator"][simulator_name]["world_range"])
        self.ground_resolution = np.array(cfg["mapper"]["ground_resolution"])
        self.num_dimensions = len(self.map_boundary)

        if use_torch:
            self.map_boundary = torch.Tensor(cfg["mapper"]["map_boundary"])
            self.world_range = torch.Tensor(cfg["simulator"][simulator_name]["world_range"])
            self.ground_resolution = torch.Tensor(cfg["mapper"]["ground_resolution"])

        (
            self.terrain_map,
            self.epistemic_map,
            self.occupancy_map,
            self.hit_map_occupancy,
            self.hit_map_semantic,
            self.train_data_map,
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
        raise NotImplementedError("'init_map' function not implemented!")

    @property
    def class_num(self) -> int:
        task = self.cfg["simulator"]["task"]
        if task == "classification":
            return self.cfg["mapper"]["class_number"]
        elif task == "regression":
            return 1
        else:
            raise NotImplementedError(f"Mapping for {task} task is not implemented!")

    @property
    def representation_prior_const(self) -> float:
        return 0.7

    @property
    def uncertainty_prior_const(self) -> float:
        task = self.cfg["simulator"]["task"]
        if task == "classification":
            if self.model_cfg["model"]["evidential_model"]:
                return 0.3

            use_entropy_criterion = (
                self.model_cfg["train"]["num_mc_epistemic"] <= 1 and not self.model_cfg["model"]["ensemble_model"]
            )
            if use_entropy_criterion:
                return np.log(self.class_num)

            return 0.2
        elif task == "regression":
            return self.terrain_map.variance_prior_const
        else:
            raise NotImplementedError(f"Mapping for {task} task is not implemented!")

    def update_map(self, data_source: Dict):
        raise NotImplementedError("'update_map' function not implemented!")

    def get_map_state(
        self, pose: np.array, depth: np.array = None, sensor: Sensor = None
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        raise NotImplementedError("'get_map_state' function not implemented!")
