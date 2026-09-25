# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
uv sync && source .venv/bin/activate && uv pip install -e .
```

## Common Commands

**Run tests:**
```bash
pytest src/
```

**Lint and format:**
```bash
isort .
ruff format . && ruff check --fix
```

**Type checking:**
```bash
mypy src/
```

**Run standard ZO training:**
```bash
uv run src/zo_main.py --config-path text_classification/gaussian_opt.yaml
```

**Run dynamics experiment (SGD vs ZO comparison):**
```bash
uv run src/llm_dynamics_main.py --config-path text_classification/gaussian_opt.yaml --num-test-samples 5
```

**Run all dynamics experiments in batch:**
```bash
bash src/run_all_dynamics_experiments.sh --num-test-samples 5
```

**Inspect dataset without training:**
```bash
uv run src/print_test_samples.py --config-path text_classification/gaussian_opt.yaml --num-test-samples 5
```

## Architecture Overview

This is a research project studying **zeroth-order (ZO) optimization dynamics for LLM fine-tuning**, comparing ZO gradient estimation to standard SGD.

### Core Components

**Entry points (`src/`):**
- `zo_main.py` — Standard ZO training pipeline: loads config → model → trains with `ZOOptimizer`
- `llm_dynamics_main.py` — Comparative experiment tracking logit/probability trajectories across training steps for both SGD and ZO on the same test samples; outputs JSON to `results/text_classification/`
- `run_all_dynamics_experiments.sh` — Batch runner for all configs under `configs/text_classification/dynamics_experiments/`

**ZO optimization (`src/zo_llm/`):**
- `zo_optim.py` — `ZOOptimizer` implementing forward/central ZO gradient estimation with pluggable perturbation distributions (Gaussian, Bernoulli, Uniform, Randomized Gaussian, etc.)
- `perturb.py` — `PerturbBase` abstract class + concrete perturbation strategy implementations
- `llm_trainer.py` — `LLM_trainer` wrapping the ZO training loop; `train_one_step()` calls the optimizer, `eval_model()` computes loss/accuracy

**Utilities (`src/zo_llm/util/`):**
- `config_parser.py` — Attrs-based `MyConfig` dataclass; YAML → structured config with enums for `LargeModel` and `RandomGradEstimateMethod`
- `language_utils.py` — `CustomLMDataset` / `CustomLMGenerationDataset`, prompt `Template` hierarchy (SST2, QQP, RTE, etc.), `LLMBatchInput`, and loss functions (`get_lm_loss()` with modes: `last_token`, `accuracy`, `f1`, `full_sentence`)
- `data_utils.py` — `get_dataloaders()` for classification (SST2, QQP, RTE, MRPC) and generation (SQuAD, DROP, XSUM) tasks
- `prepare_settings.py` — `get_model()` loading HuggingFace models (OPT, GPT2, Llama, Phi, DeepSeek-Qwen) with dropout=0; `get_model_inferences_and_metrics()` returning task-specific inference/metric functions

### Config System

YAML configs live in `src/zo_llm/configs/text_classification/`. Key fields: `device`, `large_model`, `lr`, `iterations`, `pert_distribution`, `num_pert`, `mu`, `seed`. The `dynamics_experiments/` subdirectory contains 36+ configs varying model sizes and perturbation counts.

### Results

- ZO training: TensorBoard logs → `results/zo_llm/{dataset}/{timestamp}/`
- Dynamics experiments: JSON with per-sample logit/probability histories → `results/text_classification/`
