import torch
from datasets import concatenate_datasets
from datasets import load_dataset as huggingface_load_dataset

from zo_llm.util.config_parser import MyConfig
from zo_llm.util.language_utils import (
    LM_DATASET_CONFIG_MAP,
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


def get_dataloaders(
    data_setting: MyConfig, seed: int, hf_model_name: str | None = None
) -> tuple[
    list[torch.utils.data.DataLoader],
    torch.utils.data.DataLoader,
]:
    if data_setting.dataset == LmClassificationTask.sst2:
        max_length = 32
    elif data_setting.dataset == LmClassificationTask.sst5:
        max_length = 64
    elif data_setting.dataset == LmClassificationTask.yahoo_answers:
        max_length = 128
    else:
        max_length = 2048

    if isinstance(data_setting.dataset, LmClassificationTask):
        dataset_name = LM_DATASET_MAP[data_setting.dataset.value]
        config_name = LM_DATASET_CONFIG_MAP[data_setting.dataset.name]
        dataset = (
            huggingface_load_dataset(dataset_name, config_name)
            if config_name
            else huggingface_load_dataset(dataset_name)
        )
        if data_setting.dataset == LmClassificationTask.yahoo_answers:
            # Remap 1-indexed topic (1–10) to 0-indexed label (0–9)
            def remap_label(sample):
                sample["label"] = sample["class_index"] - 1
                return sample

            n_classes = 10
            train_splits = []
            test_splits = []
            train_by_class = dataset["train"].map(remap_label)
            test_by_class = dataset["test"].map(remap_label)
            for cls in range(n_classes):
                train_splits.append(
                    train_by_class.filter(lambda x, c=cls: x["label"] == c).select(range(1000))
                )
                test_splits.append(
                    test_by_class.filter(lambda x, c=cls: x["label"] == c).select(range(100))
                )
            raw_train_dataset = concatenate_datasets(train_splits)
            raw_test_dataset = concatenate_datasets(test_splits)
        else:
            raw_train_dataset = dataset["train"]
            raw_test_dataset = dataset["validation"]
        tokenizer = get_hf_tokenizer(hf_model_name)
        template = LM_TEMPLATE_MAP[data_setting.dataset.value]()
        encoded_train_texts = list(map(template.verbalize, raw_train_dataset))
        encoded_test_texts = list(map(template.verbalize, raw_test_dataset))
        # Convert raw datasets to lists for storage
        raw_train_samples = [raw_train_dataset[i] for i in range(len(raw_train_dataset))]
        raw_test_samples = [raw_test_dataset[i] for i in range(len(raw_test_dataset))]
        train_dataset = CustomLMDataset(
            encoded_train_texts, raw_train_dataset["label"], tokenizer, max_length=max_length,
            raw_samples=raw_train_samples
        )
        test_dataset = CustomLMDataset(
            encoded_test_texts, raw_test_dataset["label"], tokenizer, max_length=max_length,
            raw_samples=raw_test_samples
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=data_setting.train_batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=get_collate_fn(tokenizer, max_length),
        )
        test_generator = torch.Generator().manual_seed(seed)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=data_setting.test_batch_size,
            shuffle=True,
            generator=test_generator,
            collate_fn=get_collate_fn(tokenizer, max_length),
        )
    elif isinstance(data_setting.dataset, LmGenerationTask):
        dataset = huggingface_load_dataset(LM_DATASET_MAP[data_setting.dataset.value])
        raw_train_dataset = dataset["train"].select(range(1000)).shuffle(seed)
        raw_test_dataset = dataset["validation"].select(range(100)).shuffle(seed)
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
        # Convert raw datasets to lists for storage
        raw_train_samples = [raw_train_dataset[i] for i in range(len(raw_train_dataset))]
        raw_test_samples = [raw_test_dataset[i] for i in range(len(raw_test_dataset))]
        train_dataset = CustomLMDataset(
            encoded_train_texts, labels=None, tokenizer=tokenizer, max_length=max_length, raw_samples=raw_train_samples
        )
        test_dataset = CustomLMGenerationDataset(
            encoded_test_texts, test_golds, tokenizer, max_length=max_length,
            raw_samples=raw_test_samples
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=data_setting.train_batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=get_collate_fn(tokenizer, max_length),  # Notice the collate_fn
        )
        test_generator = torch.Generator().manual_seed(seed)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=data_setting.test_batch_size,
            shuffle=True,
            generator=test_generator,
            collate_fn=get_collate_fn_for_gen_model(tokenizer, max_length),
        )

    return train_loader, test_loader
