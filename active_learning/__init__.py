from typing import Dict, Optional

from active_learning.learners import ModelLearner, Learner


def get_learner(model_cfg: Dict, weights_path: Optional[str], logger_name: str, model_id: str = "0") -> Learner:
    return ModelLearner(model_cfg, weights_path, logger_name, model_id=model_id)
