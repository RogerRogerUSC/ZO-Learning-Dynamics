# ZO-LLM

## Quick Start

We use `uv` to manage the enviroment. After installing the `uv` in your system, run the following
commands to setup the enviroments.

```py
uv sync
source .venv/bin/activate
```

We use `ruff` to format and check the lint.
```py
ruff format .
ruff check --fix
```