from enum import Enum
from functools import cached_property

import torch
from torch.utils.data.dataset import Subset, Dataset
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset as huggingface_load_dataset

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict

from util.language_utils import (
    LM_DATASET_MAP,
    LM_TEMPLATE_MAP,
    CustomLMDataset,
    CustomLMGenerationDataset,
    LmClassificationTask,
    LmGenerationTask,
    get_collate_fn,
    get_collate_fn_for_gen_model,
    get_hf_tokenizer,
)


class DataSetting(BaseSettings, cli_parse_args=True, cli_ignore_unknown_args=True):
    # pydantic's config, not neural network model
    model_config = SettingsConfigDict(frozen=True)

    # data
    dataset: LmClassificationTask | LmGenerationTask = Field(default=LmClassificationTask.sst2)
    train_batch_size: int = Field(default=8, validation_alias=AliasChoices("train-batch-size"))
    test_batch_size: int = Field(default=8, validation_alias=AliasChoices("test-batch-size"))

    @cached_property
    def data_setting(self):
        return DataSetting()


def get_dataloaders(
    data_setting: DataSetting, seed: int, hf_model_name: str | None = None
) -> tuple[
    list[torch.utils.data.DataLoader],
    torch.utils.data.DataLoader,
]:
    if isinstance(data_setting.dataset, (LmClassificationTask, LmGenerationTask)):
        assert isinstance(hf_model_name, str)
        if data_setting.dataset == LmClassificationTask.sst2:
            max_length = 32
        else:
            max_length = 2048

        if isinstance(data_setting.dataset, LmClassificationTask):
            dataset = huggingface_load_dataset(
                LM_DATASET_MAP[data_setting.dataset.value], data_setting.dataset.value
            )
            raw_train_dataset = dataset["train"]
            raw_test_dataset = dataset["test"]
            tokenizer = get_hf_tokenizer(hf_model_name)
            template = LM_TEMPLATE_MAP[data_setting.dataset.value]()
            encoded_train_texts = list(map(template.verbalize, raw_train_dataset))
            encoded_test_texts = list(map(template.verbalize, raw_test_dataset))
            train_dataset = CustomLMDataset(encoded_train_texts, tokenizer, max_length=max_length)
            test_dataset = CustomLMDataset(encoded_test_texts, tokenizer, max_length=max_length)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=data_setting.train_batch_size,
                shuffle=True,
                collate_fn=get_collate_fn(tokenizer, max_length),
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=data_setting.test_batch_size,
                shuffle=True,
                collate_fn=get_collate_fn(tokenizer, max_length),
            )
        elif isinstance(data_setting.dataset, LmGenerationTask):
            dataset = huggingface_load_dataset(LM_DATASET_MAP[data_setting.dataset.value])
            raw_train_dataset = dataset["train"].select(range(1000)).shuffle(seed)
            raw_test_dataset = dataset["test"].select(range(100)).shuffle(seed)
            tokenizer = get_hf_tokenizer(hf_model_name)
            template = LM_TEMPLATE_MAP[data_setting.dataset.value]()
            # Notice the difference between train and test dataset preparation.
            # "verbalize" function generates text including the answers
            # "encode" function generates text without the answers
            encoded_train_texts = list(map(template.verbalize, raw_train_dataset))
            encoded_test_texts = list(map(template.encode, raw_test_dataset))
            if data_setting.dataset == LmGenerationTask.squad:
                test_golds = list(map(lambda d: d["answers"]["text"][0], raw_test_dataset))
            elif data_setting.dataset == LmGenerationTask.drop:
                test_golds = list(map(lambda d: d["answers_spans"]["spans"][0], raw_test_dataset))
            elif data_setting.dataset == LmGenerationTask.xsum:
                test_golds = list(map(lambda d: d["summary"], raw_test_dataset))
            train_dataset = CustomLMDataset(encoded_train_texts, tokenizer, max_length=max_length)
            test_dataset = CustomLMGenerationDataset(
                encoded_test_texts, test_golds, tokenizer, max_length=max_length
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=data_setting.train_batch_size,
                shuffle=True,
                collate_fn=get_collate_fn(tokenizer, max_length),  # Notice the collate_fn
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=data_setting.test_batch_size,
                shuffle=True,
                collate_fn=get_collate_fn_for_gen_model(tokenizer, max_length),
            )
    else:
        raise Exception(f"Dataset {data_setting.dataset} is not supported!")

    return train_loader, test_loader
