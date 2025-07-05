from pathlib import Path
import yaml
import attrs
import torch
from enum import Enum

from util.language_utils import SUPPORTED_LLM, LmClassificationTask

file_path = Path(__file__)


class RandomGradEstimateMethod(Enum):
    rge_central = "rge-central"
    rge_forward = "rge-forward"


class LargeModel(Enum):
    opt_125m = "opt-125m"
    opt_350m = "opt-350m"
    opt_1p3b = "opt-1.3b"
    opt_2p7b = "opt-2.7b"
    opt_6p7b = "opt-6.7b"
    opt_13b = "opt-13b"
    opt_30b = "opt-30b"
    deepseek_qwen_1p5b = "deepseek-qwen-1.5b"


# Step 1: Define your config dataclass
@attrs.mutable
class MyConfig:
    # General
    device: str
    large_model: LargeModel = attrs.field(converter=LargeModel)
    model_dtype: str

    # training parameters
    lr: float
    iterations: int
    eval_iterations: int

    # dataset
    dataset: LmClassificationTask = attrs.field(converter=LmClassificationTask)
    # dataset: LmGenerationTask = attrs.field(converter=LmGenerationTask)
    train_batch_size: int
    test_batch_size: int

    # zo parameters
    estimator_type: str
    grad_estimate_method: RandomGradEstimateMethod = attrs.field(converter=RandomGradEstimateMethod)
    num_pert: int
    mu: float
    pert_distribution: str

    # MISC
    seed: int
    log_to_tensorboard: str | None = None

    def get_hf_model_name(self):
        return SUPPORTED_LLM[self.large_model.value]

    def get_torch_dtype(self):
        return {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[self.model_dtype]

    def get_device(self):
        return torch.device(self.device)


# Step 2: YAML Loader
def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Step 3: Parse into dataclass
def parse_config(filename: str) -> MyConfig:
    config_dict = load_yaml_config(file_path.parent / ".." / "configs" / filename)
    return MyConfig(**config_dict)
