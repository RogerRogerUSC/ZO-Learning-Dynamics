# ZO-LLM

## Quick Start

We use `uv` to manage the enviroment. After installing the `uv` in your system, run the following
commands to setup the enviroments.

```py
uv sync
source .venv/bin/activate
uv pip install -e .
```

We use `ruff` to format and check the lint.
```py
isort .
ruff format .
ruff check --fix
```