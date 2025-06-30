from typing import Literal
from functools import cached_property

import torch
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict
from exp_helper.exp_typing import LargeModel, ModelDtype
from enum import Enum
from grad_estimators.random_grad_estimator import RandomGradEstimateMethod
from util.language_utils import SUPPORTED_LLM
from exp_helper.data import DataSetting  # noqa: F401, for a central export place for all settings


class FrozenSetting(BaseSettings, cli_parse_args=True, cli_ignore_unknown_args=True):
    # pydantic's config, not neural network model
    model_config = SettingsConfigDict(frozen=True)


class GeneralSetting(FrozenSetting):
    # general
    seed: int = Field(
        default=365,
        description="Random seed used to initialize model, get dataloaders and to sample the RGE's seeds each round",
    )
    log_to_tensorboard: str | None = Field(
        default=None,
        validation_alias=AliasChoices("log-to-tensorboard"),
        description="Provide a valid path, will log training process to the tensorboard in that path",
    )

    @cached_property
    def general_setting(self) -> "GeneralSetting":
        return GeneralSetting()


class DeviceSetting(FrozenSetting):
    cuda: CliImplicitFlag[bool] = Field(
        default=True, description="--no-cuda will disable cuda training"
    )
    mps: CliImplicitFlag[bool] = Field(
        default=True,
        description="--no-mps will disable macOS GPU training, this command line argument is ignored when cuda is available and choose to use cuda",
    )

    @cached_property
    def device_setting(self) -> "DeviceSetting":
        return DeviceSetting()


class ModelSetting(FrozenSetting):
    # model
    large_model: LargeModel = Field(
        default=LargeModel.opt_125m,
        validation_alias=AliasChoices("large-model"),
        description="Model name for Hugging Face Lanuguage Model. current only support facebook/opt families",
    )
    model_dtype: ModelDtype = Field(
        default=ModelDtype.float32, validation_alias=AliasChoices("model-dtype")
    )

    # LoRA
    lora: CliImplicitFlag[bool] = Field(default=False)
    lora_r: int = Field(default=8, validation_alias=AliasChoices("lora-r"))
    lora_alpha: int = Field(default=16, validation_alias=AliasChoices("lora-alpha"))

    @cached_property
    def model_setting(self) -> "ModelSetting":
        return ModelSetting()

    def get_hf_model_name(self) -> str:
        return SUPPORTED_LLM[self.large_model.value]

    def get_torch_dtype(self):
        return {
            ModelDtype.float16: torch.float16,
            ModelDtype.float32: torch.float32,
            ModelDtype.bfloat16: torch.bfloat16,
        }[self.model_dtype]


class OptimizerSetting(FrozenSetting):
    # optimizer
    optimizer: Literal["sgd", "adam"] = Field(default="sgd")
    lr: float = Field(default=1e-4)
    momentum: float = Field(default=0)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)

    @cached_property
    def optimizer_setting(self) -> "OptimizerSetting":
        return OptimizerSetting()


class EstimatorType(Enum):
    vanilla = "vanilla"


class RGESetting(FrozenSetting):
    estimator_type: EstimatorType = Field(
        default=EstimatorType.vanilla,
        validation_alias=AliasChoices("estimator-type"),
        description="Type of gradient estimator, options: vanilla",
    )
    mu: float = Field(default=1e-3, description="Perturbation step to measure local gradients")
    num_pert: int = Field(
        default=1,
        validation_alias=AliasChoices("num-pert"),
        description="Number of perturbations needed to perform when estimating gradient",
    )
    adjust_perturb: CliImplicitFlag[bool] = Field(
        default=False,
        validation_alias=AliasChoices("adjust-perturb"),
        description="Whether to adjust number of perturbation in the training process",
    )
    grad_estimate_method: RandomGradEstimateMethod = Field(
        default=RandomGradEstimateMethod.rge_central,
        validation_alias=AliasChoices("grad-estimate-method"),
        description="Forward or Central",
    )
    optim: CliImplicitFlag[bool] = Field(
        default=True,
        description="Use optimizer or not, when no-optim, update model without torch.optim (SGD only). This can significantly save memory.",
    )

    @cached_property
    def rge_setting(self) -> "RGESetting":
        return RGESetting()


class NormalTrainingLoopSetting(FrozenSetting):
    iterations: int = Field(default=500)
    warmup_iterations: int = Field(default=5, validation_alias=AliasChoices("warmup-iterations"))
    eval_iterations: int = Field(default=10, validation_alias=AliasChoices("eval-iterations"))

    @cached_property
    def normal_training_loop_setting(self) -> "NormalTrainingLoopSetting":
        return NormalTrainingLoopSetting()
