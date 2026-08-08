# Proposed Repository Structure

The public repository should present the project as a clean research system,
not as a full working directory containing every intermediate experiment.

## Target Layout

```text
configs/
  deployment/          Hardware deployment configs and GUI templates
  experiments/         Reproducible training configs
  baselines/           Baseline comparison configs

control/               Real-time deployment loop and hardware I/O protocol
core/                  RL environment, SAC agent, CORAL/SafetyNet, SoH inference
gui/                   Deployment GUI source
packaging/             Build specifications and release scripts
tests/                 Unit and regression tests

data/
  scripts/
    preprocessing/     Raw-log conversion and dataset builders
    diagnostics/       Sensor freeze, PV/load, and deployment anomaly checks
    figures/           Report-ready plotting scripts
    baselines/         Baseline generation and summary utilities
    archive/           Older one-off scripts kept for traceability
  raw/.gitkeep         Placeholder only; raw logs stay local or external

docs/
  methods/             Method descriptions and experiment protocols
  deployment/          Deployment notes and hardware behavior summaries
  figures/             A small curated set of final figures, if needed

examples/              Self-contained examples without large generated outputs
```

## Current Cleanup Priorities

1. **Keep the top-level directory minimal.**
   Top-level files should be limited to `README.md`, `LICENSE`, dependency
   files, and a small number of build/deployment entry points.

2. **Reduce config sprawl.**
   Many `config_p302_v*.yaml` files are historical experiments. Move them into
   an archive or group them by purpose:
   `configs/experiments/`, `configs/deployment/`, `configs/baselines/`.

3. **Separate reproducible scripts from one-off analysis.**
   Keep reusable scripts grouped under `data/scripts/preprocessing/`,
   `data/scripts/diagnostics/`, `data/scripts/figures/`, and
   `data/scripts/baselines/`. Keep older one-off scripts in
   `data/scripts/archive/` only when they still help trace the research history.

4. **Do not track raw logs or generated figures by default.**
   Raw deployment CSVs and generated PNG/PDF reports should not be part of the
   public branch unless they are a small curated sample.

5. **Curate examples.**
   Example environments are useful for portfolio value, but generated result
   folders and `.zip` model outputs should stay out of Git.

6. **English public surface first.**
   Prioritize English for `README.md`, `docs/`, CLI help, and config comments.
   Internal comments can be translated gradually.

## Suggested First Public Commit

The first cleanup commit should include:

- rewritten `README.md`
- repository hygiene and structure docs
- improved `.gitignore`
- SoH online inference code
- reusable deployment and diagnostic scripts
- tests and config templates

It should exclude:

- raw deployment logs
- generated figures
- packaged GUI releases
- large RL checkpoints
- local vendor paths
