# Plan: Add Yahoo Answers 10-Category Dataset

## Overview
Add `yassiracharki/Yahoo_Answers_10_categories_for_NLP` as a classification task using `question_title` and `topic` (class index). The dataset has 10 categories with 1-indexed labels that need normalizing to 0-indexed. Training set: 1000 samples per class (10k total), test set: 100 samples per class (1k total), sampled from HuggingFace `train` and `test` splits respectively. A base config and 30 dynamics experiment configs (3 models × 2 distributions × 5 num_pert values: 1, 5, 10, 20, 50) are created.

## Files to Change

### `src/zo_llm/util/language_utils.py`
- Add `yahoo_answers = "yahoo_answers"` to `LmClassificationTask` enum
- Add `YahooAnswersTemplate(ClassificationTemplate)` with a 10-class verbalizer:
  `{0: " society", 1: " science", 2: " health", 3: " education", 4: " computer", 5: " sports", 6: " business", 7: " entertainment", 8: " family", 9: " politics"}`
- `verbalize_for_pred` uses: `f"Question: {sample['question_title']}\nCategory:"`
- Register in `LM_DATASET_MAP`: `"yahoo_answers": "yassiracharki/Yahoo_Answers_10_categories_for_NLP"`
- Register in `LM_DATASET_CONFIG_MAP`: `"yahoo_answers": None`
- Register in `LM_TEMPLATE_MAP`: `"yahoo_answers": YahooAnswersTemplate`

### `src/zo_llm/util/data_utils.py`
- Add `max_length = 128` branch for `yahoo_answers`
- Add Yahoo Answers-specific loading block inside the `LmClassificationTask` branch:
  - Load from `train` split (for training) and `test` split (for test) — no `validation` split
  - Remap `topic` field (1-indexed 1–10) → `label` field (0-indexed 0–9) via `dataset.map()`
  - Subsample: 1000 samples per class (10k) for train, 100 per class (1k) for test, maintaining class balance

## Files to Create

### `src/zo_llm/configs/text_classification/yahoo_answers_gaussian_opt.yaml`
- Base config: `opt-125m`, gaussian, `num_pert: 5`, `dataset: "yahoo_answers"`, `iterations: 3000`

### `src/zo_llm/configs/text_classification/dynamics_experiments/yahoo_answers/{model}_{dist}_{n}.yaml` (30 files)
- 3 models: `opt-125m`, `opt-350m`, `opt-1.3b`
- 2 distributions: `gaussian`, `bernoulli`
- 5 num_pert values: `1`, `5`, `10`, `20`, `50` (exclude 100)
- All with `iterations: 200`, `dataset: "yahoo_answers"`, matching the sst2/sst5 dynamics config style

## Steps
1. Edit `language_utils.py`: add enum entry, `YahooAnswersTemplate`, and three map registrations
2. Edit `data_utils.py`: add `max_length` case and Yahoo Answers-specific loading (split + remap + subsample)
3. Create base YAML config
4. Create 30 dynamics experiment YAML configs under `dynamics_experiments/yahoo_answers/`
