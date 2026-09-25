<h1 align="center">Learning Dynamics of Zeroth-Order Optimization:<br>A Kernel Perspective</h1>

<p align="center">
	<img alt="ICML 2026" src="https://img.shields.io/badge/ICML-2026-176B5B?style=for-the-badge">
	<img alt="Proceedings of Machine Learning Research, volume 306" src="https://img.shields.io/badge/PMLR-Volume%20306-344E73?style=for-the-badge">
</p>

<p align="center">
	<strong>Zhe Li · Bicheng Ying · Zidong Liu · Haibo Yang</strong><br>
	Proceedings of the 43rd International Conference on Machine Learning<br>
	Seoul, South Korea · 2026
</p>

<p align="center">
	<a href="https://openreview.net/forum?id=UTIylOfVUb">OpenReview</a> ·
	<a href="https://arxiv.org/abs/2605.03373">arXiv</a> ·
	<a href="#quick-start">Quick start</a> ·
	<a href="#experiments">Experiments</a>
</p>

---

This repository contains the training code and experiment workflows accompanying the paper. The central question is why zeroth-order (ZO) optimization can remain effective for very large models despite classical dimension-dependent convergence bounds.

## The Kernel View

The paper expresses the ZO empirical neural tangent kernel (eNTK) as a random projection of the first-order (FO) kernel:

$$
K_t^{ZO}(x_o, x_u) = J_t(x_o)^\top U_{t,P} U_{t,P}^\top J_t(x_u),
$$

where $J_t(x)$ is the parameter Jacobian of the model output and $U_{t,P}$ contains $P$ normalized perturbation directions. This links ZO learning dynamics to the Johnson–Lindenstrauss lemma.

| Paper insight | What it means |
| --- | --- |
| **Perturbation budget $P$** | More directions improve the fidelity of the projected kernel. |
| **Output size $V$** | The analysis depends on the model's output dimension. |
| **Parameter count $d$** | The kernel approximation bound does not scale directly with the number of parameters. |

## Quick Start

Requires Python 3.10 or newer. Install [`uv`](https://docs.astral.sh/uv/) and set up the environment:

```bash
uv sync
uv pip install -e .
```

The code uses PyTorch, Transformers, Hugging Face Datasets, and Matplotlib. Choose a PyTorch build that supports your hardware. Configuration presets select either CUDA (`cuda`) or Apple Silicon (`mps`); update the YAML `device` field when needed.

## Train an LLM

Run training commands from the repository root. Config paths are relative to `src/zo_llm/configs/`.

**SST-2 with uniform perturbations:**

```bash
uv run python src/zo_main.py --config-path text_classification/uniform_opt.yaml
```

**SST-2 with Bernoulli/Rademacher perturbations:**

```bash
uv run python src/zo_main.py --config-path text_classification/bernoulli_opt.yaml
```

<!-- **SQuAD generation:**

```bash
uv run python src/zo_main.py --config-path text_generation.yaml
``` -->

The selected dataset, tokenizer, and model are loaded through Hugging Face Datasets and Transformers; model files may download on first use. Training and evaluation metrics are printed to the terminal. Setting `log_to_tensorboard` in a config writes scalar logs below `results/zo_llm/`.

### Configuring an Experiment

Presets live in `src/zo_llm/configs/`. They specify model and task, device and dtype, learning rate, iteration count, batch sizes, estimator, perturbation distribution, perturbation count, and smoothing parameter `mu`.

Implemented classification templates include SST-2, QQP, RTE, MultiRC, CB, WiC, WSC, and BoolQ. Generation templates include SQuAD, DROP, and XSum. Model aliases and their Hugging Face IDs are listed in `src/zo_llm/util/language_utils.py`.

The optimizer supports Gaussian, Bernoulli/Rademacher, uniform, and randomized Gaussian perturbations. Choose `rge-forward` or `rge-central` for the gradient-estimation method.

## Experiments

The paper's analysis workflows are collected under `src/zo_dynamics/`:

| Notebook or script | Experiment |
| --- | --- |
| `linear_func.ipynb` | Kernel behavior on linear functions. |
| `mnist_example.ipynb` | FO and projected ZO eNTK comparisons; error versus perturbation count. |
| `llm_data.ipynb` | Frobenius-error summaries and LLM prediction trajectories. |
| `mnist_example.py` | LeNet/MNIST training and dynamics example. |

Open the notebooks in VS Code with the project environment selected as the kernel. Set the notebook working directory to `src/zo_dynamics`, since the notebooks use paths relative to that directory. Generated PDFs are written to [`src/zo_dynamics/figures/`](src/zo_dynamics/figures/).

**Data and checkpoints**

- The MNIST notebook downloads MNIST when needed, but expects `checkpoints/mnist_example.pth` at the repository root. The checkpoint is not included and must be supplied separately.
- The LLM trajectory notebook reads the JSON experiment files in `llm_results/`.
- Training logs and experiment result files are stored under `results/`.

## Development

Format and lint with Ruff:

```bash
uv run ruff format .
uv run ruff check .
```

## Citation

```bibtex
@inproceedings{
    li2026learning,
    title={Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective},
    author={Zhe Li and Bicheng Ying and Zidong Liu and Haibo Yang},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=UTIylOfVUb}
}
```