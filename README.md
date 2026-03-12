## PSO Feature Selection Convergence (Mealpy)

This repo runs a **Particle Swarm Optimization (PSO)** wrapper-based feature selection experiment (with **Mealpy**) on `class.csv`, then saves a convergence-style plot to `pso_feature_selection_convergence.png`.

## Requirements

- **Python**: \(>= 3.13\) (per `pyproject.toml`)
- **uv**: Python package manager / runner

## Setup (uv)

### macOS

Install uv (recommended):

```bash
brew install uv
```

Alternative (official installer):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

Install uv:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Close and reopen your terminal after installing so `uv` is on PATH.

## Run

From the repo root (where `pyproject.toml` is), run:

```bash
uv run plot_pso_fs_convergence.py
```

Expected outputs:

- Reads: `class.csv`
- Writes: `pso_feature_selection_convergence.png`
- Prints progress logs like `[PSO-FS] ...`

## Notes

- If you have multiple Python versions installed, uv will pick a compatible one automatically; if it can’t find Python \(>= 3.13\), install Python 3.13 and retry.
- To re-run from a clean state, just run the same `uv run ...` command again (uv will reuse the environment).
# fs-pso
