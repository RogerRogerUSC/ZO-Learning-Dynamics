import torch
from typing import Callable, TypeAlias
from functools import partial
from transformers import AutoModelForCausalLM
from util import model_helpers
from util.model_helpers import AllModel
from grad_estimators.random_grad_estimator import RandomGradientEstimator
from util.language_utils import (
    LM_TEMPLATE_MAP,
    SUPPORTED_LLM,
    get_lm_loss,
    get_hf_tokenizer,
)
from exp_helper.config_parser import MyConfig
from exp_helper.data import LmClassificationTask, LmGenerationTask
from dataclasses import dataclass


SupportedDataset: TypeAlias = LmClassificationTask | LmGenerationTask


def get_model(
    dataset: SupportedDataset,
    model_setting: MyConfig,
    seed: int | None = None,
) -> AllModel:
    torch_dtype = model_setting.get_torch_dtype()
    model: AllModel
    if seed:
        torch.manual_seed(seed)

    if isinstance(dataset, (LmClassificationTask, LmGenerationTask)):
        assert model_setting.large_model.value in SUPPORTED_LLM
        hf_model_name = model_setting.get_hf_model_name()
        model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch_dtype)
        model.model_name = model_setting.large_model.value
        # if model_setting and model_setting.lora:
        #     # this step initialize lora parameters, which should be under control of seed
        #     lora_config = LoraConfig(
        #         r=model_setting.lora_r,
        #         lora_alpha=model_setting.lora_alpha,
        #         target_modules=["q_proj", "v_proj"],
        #     )
        #     model = get_peft_model(model, lora_config).to(torch_dtype)
        return model
    else:
        raise Exception(f"Dataset {dataset} is not supported")


def get_optimizer(
    model: AllModel, dataset: SupportedDataset, optimizer_setting: MyConfig
) -> torch.optim.SGD:
    trainable_model_parameters = model_helpers.get_trainable_model_parameters(model)
    if isinstance(dataset, LmClassificationTask):
        return torch.optim.SGD(
            trainable_model_parameters,
            lr=optimizer_setting.lr,
            momentum=0,
            weight_decay=5e-4,
        )
    elif isinstance(dataset, LmGenerationTask):
        return torch.optim.SGD(
            trainable_model_parameters,
            lr=optimizer_setting.lr,
            momentum=0,
            weight_decay=0,
        )
    else:
        raise Exception(f"dataset {dataset.value} not supported")


@dataclass
class ModelInferences:
    train_inference: Callable
    test_inference: Callable


@dataclass
class MetricPacks:
    train_loss: Callable
    train_acc: Callable
    test_loss: Callable
    test_acc: Callable


def get_model_inferences_and_metrics(
    dataset: SupportedDataset, model_setting: MyConfig
) -> tuple[ModelInferences, MetricPacks]:
    hf_model_name = model_setting.get_hf_model_name()
    tokenizer = get_hf_tokenizer(hf_model_name)
    if isinstance(dataset, LmGenerationTask):
        generation_kwargs = {
            "do_sample": True,
            "temperature": 1.0,
            "num_beams": 2,
            "top_k": None,
            "num_return_sequences": 1,
            "max_new_tokens": 5,  # will be adjusted dynamically later, 500 for xsum, not sure why we need it tho
            "max_length": 2048,
            "early_stopping": True,
            "eos_token_id": [
                tokenizer.encode("\n", add_special_tokens=False)[-1],
                tokenizer.eos_token_id,
            ],
            "top_p": 0.95 if dataset == "xsum" else 0.3,
            "length_penalty": 1 if dataset == "xsum" else 2,
        }
        # write in separate lines to differentiate from above cases, here acc=criterion
        train_criterion = get_lm_loss("full_sentence", verbalizer_id_map={})
        test_accuracy_func = get_lm_loss("f1", tokenizer=tokenizer)
        return (
            ModelInferences(
                train_inference=model_helpers.model_forward,
                test_inference=partial(
                    model_helpers.model_generate, generation_kwargs=generation_kwargs
                ),
            ),
            MetricPacks(
                train_loss=train_criterion,
                train_acc=lambda pred, true: torch.tensor(0.0),  # noop training acc step here
                test_loss=lambda pred, true: torch.tensor(0.0),  # noop test loss step here
                test_acc=test_accuracy_func,
            ),
        )
    else:
        template = LM_TEMPLATE_MAP[dataset.value]()
        verbalizer_id_map = template.get_verbalizer_id(tokenizer)  # type: ignore[attr-defined]
        train_criterion = test_criterion = get_lm_loss(
            "last_token", verbalizer_id_map=verbalizer_id_map
        )
        train_accuracy_func = test_accuracy_func = get_lm_loss(
            "accuracy", verbalizer_id_map=verbalizer_id_map
        )
        return ModelInferences(
            model_helpers.model_forward, model_helpers.model_forward
        ), MetricPacks(
            train_loss=train_criterion,
            train_acc=train_accuracy_func,
            test_loss=test_criterion,
            test_acc=test_accuracy_func,
        )


def get_gradient_estimator(
    model: AllModel, device: torch.device, config: MyConfig
) -> RandomGradientEstimator:
    if config.estimator_type == "vanilla":
        return RandomGradientEstimator(
            parameters=model_helpers.get_trainable_model_parameters(model),
            mu=config.mu,
            num_pert=config.num_pert,
            grad_estimate_method=config.grad_estimate_method,
            device=device,
            torch_dtype=config.get_torch_dtype(),
        )
    else:
        raise ValueError(f"Invalid estimator type: {config.estimator_type}")
