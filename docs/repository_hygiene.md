# Repository Hygiene

This repository is linked from a resume, so the public branch should prioritize
readability, reproducibility, and source-code clarity over preserving every
intermediate artifact from the research process.

## Keep in Git

- Source code under `core/`, `control/`, `gui/`, and `data/scripts/`.
- Reusable training/deployment configurations under `configs/`.
- Lightweight documentation under `README.md`, `docs/`, and selected example
  READMEs.
- Unit and regression tests under `tests/`.
- Small model artifacts required for bundled inference, such as the SoH
  predictor weights and scaler files.
- Selected report-ready figures only when they are intentionally referenced by
  the README or documentation.

## Keep Local or External

- Raw deployment logs (`raw_data_*.csv`, `deployment_*.csv`).
- Generated figures and PDFs from exploratory analysis.
- Large trained RL checkpoints.
- PyInstaller builds, release folders, executables, and local GUI configs.
- Vendor hardware folders and runtime `Data.txt` / `Command.txt` files.
- One-off seminar drafts, temporary audits, and intermediate notebooks.

## Recommended Public Structure

```text
configs/        Reproducible experiment and deployment configs
control/        Hardware I/O and real-time control loop
core/           RL environment, agents, safety framework, SoH inference
data/scripts/   Preprocessing, diagnostics, figures, and baseline utilities
docs/           Research notes, protocols, and selected explanations
examples/       Self-contained examples that can run without private data
gui/            Deployment GUI source
tests/          Unit and regression tests
```

## Language Policy

Public-facing files should be written in English:

- `README.md`
- docs intended for GitHub readers
- config comments that explain reusable research settings
- user-facing CLI help

Internal deployment logs, local notes, or hardware operator labels may remain in
Chinese temporarily, but should not be the primary public entry point.

## Before Pushing

Run:

```bash
git status --short
pytest
```

Then review:

- Are there raw logs or generated figures accidentally staged?
- Are local Windows paths or vendor machine paths exposed?
- Does the README still explain how to navigate the repo?
- Are model artifacts small and necessary?
