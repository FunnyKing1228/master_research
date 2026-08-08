# Deployment Guide

This guide summarizes how the real-time deployment stack is expected to run on
the Windows machine connected to the P302 microgrid testbed.

## Runtime Overview

The deployment loop communicates with the vendor controller through two text
files:

```text
Vendor controller -> Data.txt -> control/io_protocol.py -> run_deployment.py
run_deployment.py -> Command.txt -> Vendor controller
```

`run_deployment.py` reads raw hardware measurements, aggregates them into
15-minute decision windows, updates the SoC tracker, evaluates the RL policy,
applies deployment guards, and writes the final command.

## Required Local Artifacts

The public repository intentionally excludes raw logs, packaged executables, and
large trained checkpoints. A deployment machine should provide:

- A vendor controller folder containing the executable and live `Data.txt` /
  `Command.txt` files.
- A trained RL checkpoint, typically `best_sac_model.pth`.
- `load_pattern.txt`, which defines the load-group schedule used by deployment.
- Optional SoH model artifacts in `soh_models/` or another selected model path.
- A writable log directory for raw readings, deployment CSVs, and SoH prediction
  outputs.

## GUI Deployment Flow

The recommended operator workflow is the Tkinter GUI in `gui/ai_control_gui.py`.

1. Select the vendor executable and verify the vendor program can start.
2. Select the RL model checkpoint.
3. Confirm `Data.txt` and `Command.txt` paths.
4. Set the initial SoC and battery PP ID.
5. Choose the current interpretation mode. The deployment default is `hybrid`.
6. Keep SoH prediction disabled unless a validated model is selected.
7. Start AI control and monitor the log/status tabs.

The GUI writes a local `config_gui.json`. This file is ignored by Git because it
contains machine-specific paths.

## Command-Line Deployment

For development or debugging, deployment can be launched directly:

```powershell
py control\run_deployment.py `
  --data-file "path\to\Data.txt" `
  --command-file "path\to\Command.txt" `
  --model-path "path\to\best_sac_model.pth" `
  --battery-pp 01 `
  --device cpu
```

Optional SoH prediction flags:

```powershell
py control\run_deployment.py `
  --data-file "path\to\Data.txt" `
  --command-file "path\to\Command.txt" `
  --model-path "path\to\best_sac_model.pth" `
  --battery-pp 01 `
  --soh-prediction `
  --soh-model-path "path\to\soh_models"
```

`--soh-use-for-capacity` should be treated as experimental. It allows predicted
SoH to adjust the effective capacity used by Coulomb counting.

## Packaging the GUI

Build the Windows release package with:

```powershell
powershell -ExecutionPolicy Bypass -File .\_deploy.ps1
```

The script:

- runs PyInstaller using `packaging/build_release.spec`,
- removes unnecessary CUDA/NVIDIA runtime files for CPU deployment,
- copies `load_pattern.txt`,
- creates a `soh_models/` folder in the release directory,
- writes a machine-local GUI config template.

The release folder is intentionally ignored by Git.

## SoH Model Folder

The release includes a dedicated `soh_models/` directory. Place lightweight SoH
model artifacts there, such as:

- `.pth` model weights,
- `.npz` scaler files,
- `.pkl` scaler files when needed.

The inference code is bundled in the repository under `core/soh_predictor/`.
Only model/scaler artifacts should be copied between machines.

## Safety Behavior

Deployment applies several guards after the raw policy action:

- low-SoC discharge blocking,
- high-SoC charge blocking,
- voltage cutoff and recovery hysteresis,
- PV-active discharge blocking,
- load-over-discharge-limit blocking,
- firmware/current inconsistency checks,
- optional SoH health lock and capacity adjustment.

The final command may differ from the raw policy action. This is expected and is
logged for replay diagnostics.

## Pre-Run Checklist

Before enabling AI control:

- Confirm the vendor controller is writing fresh `Data.txt` samples.
- Confirm `Command.txt` can be overwritten by the deployment process.
- Verify the selected model checkpoint matches the expected observation/action
  dimensions.
- Confirm the initial SoC is reasonable.
- Check that `load_pattern.txt` matches the physical load setup.
- Keep a manual stop path available for the vendor controller and GUI.

## Logs and Diagnostics

Deployment logs are written locally and should not be committed. Useful analysis
scripts live under:

```text
data/scripts/diagnostics/
data/scripts/figures/
```

Use these scripts to inspect sensor freezes, PV/load mismatch, voltage cutoff
events, SoC correction behavior, and raw-vs-final policy actions.

## Troubleshooting

- If no fresh readings appear, check the vendor controller and `Data.txt` path.
- If commands are ignored, check whether `Command.txt` is locked or overwritten
  by another process.
- If SoC drops suddenly, inspect voltage cutoff logs and firmware current
  readings before attributing the behavior to the RL policy.
- If the packaged GUI is too large, verify CUDA and unused libraries were removed
  by `_deploy.ps1`.
- If SoH prediction fails, disable SoH prediction first, then check model/scaler
  compatibility separately.
