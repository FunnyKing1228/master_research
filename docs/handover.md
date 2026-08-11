# Project Handover Notes

This document is for the next researcher or engineer who continues the P302
microgrid RL deployment work. It focuses on what is currently usable, what must
be handled carefully on hardware, and what should be treated as future work.

繁體中文生命週期交接入口：[`HANDOVER_zh.md`](HANDOVER_zh.md)。

## Current Project Status

The repository contains a deployment-safe RL control stack for a small physical
microgrid testbed. The main completed pieces are:

- SAC-based microgrid controller and P302-aligned simulation environment.
- CORAL / SafetyNet-style action correction for deployment safety.
- Real-time deployment loop using the vendor `Data.txt` / `Command.txt` protocol.
- Tkinter GUI for operator setup, model selection, deployment launch, and logs.
- Coulomb-counting SoC tracker with voltage cutoff and fallback correction.
- Deployment diagnostics for sensor freeze, voltage cutoff, and replay analysis.
- SoH and flow-rate interfaces for data collection and future model integration.

The current thesis/research scope should be understood as **deployment-safe,
hardware-aligned RL control**, not a complete electrochemical battery model.

## Main Entry Points

Use these files first:

```text
README.md                         Public project overview
docs/deployment_guide.md           How to run deployment and package the GUI
control/run_deployment.py          Main real-time deployment loop
gui/ai_control_gui.py              Operator GUI
control/io_protocol.py             Data.txt / Command.txt parser and writer
core/microgrid_env.py              Gymnasium microgrid environment
core/sac_agent.py                  SAC agent and training update logic
core/safety_net.py                 Safety projection helpers
data/scripts/diagnostics/          Deployment anomaly analysis scripts
data/scripts/figures/              Report and diagnostic plotting scripts
```

Historical experiment configs are under `configs/experiments/p302/`.
Baseline and ablation configs are under `configs/baselines/`.

## Hardware Deployment Flow

The normal hardware workflow is:

1. Start the vendor controller and confirm it updates `Data.txt`.
2. Start the GUI from the packaged release or from source.
3. Select the vendor executable.
4. Select the RL model checkpoint.
5. Select the SoH model checkpoint or model folder only if SoH prediction will
   be tested.
6. Select the log output folder for raw readings, deployment CSVs, and SoH
   prediction outputs.
7. Set initial SoC, battery PP ID, load count, and current mode.
8. Keep SoH prediction disabled unless a validated SoH model is selected.
9. Start AI control.
10. Monitor GUI logs, deployment CSVs, voltage cutoff flags, and final commands.
11. Stop AI control before stopping the vendor controller.

The command written to hardware may differ from the raw RL policy action because
deployment guards can block or project unsafe commands.

## Important Hardware Constraints

These are the main hardware lessons learned so far:

- The MOS path requires a voltage region around 7 V or higher to switch
  reliably. The battery voltage can be slightly below this region, so the vendor
  used a wiring adjustment to let the battery supply the load path.
- Battery discharge capability is small relative to some auxiliary loads.
  Current deployment constants use roughly `5.6 W` maximum discharge power and
  `8.5 W` maximum charge power for the P302 stack.
- The load bank can reach about `4 x 2.3 W = 9.2 W`.
- Flow motor power is not a small loss on the current platform. Recent
  measurements showed about `10.29 W` at 100% flow, and the motor only operates
  normally around 40-100% flow.
- Because motor power can be comparable to or larger than battery discharge
  power, the motor should be considered an auxiliary system load, not a small
  pump-efficiency correction.

Current practical recommendation:

```text
Use fixed flow for hardware deployment.
Keep the flow-rate command interface and logs.
Do not present real-time flow-rate profit optimization as completed on P302.
Treat full flow-rate optimization as future work for a larger battery module.
```

## SoC and Voltage Interpretation

The logged SoC is an operational estimate. It is mainly produced by Coulomb
counting, synthetic/current interpretation logic, and safety fallback rules. It
should not be treated as a direct measurement of true stored charge.

Deployment-log analysis showed that standby voltage near logged 20% SoC and
logged 80% SoC can be similar. This suggests that:

- the battery may not have effectively charged by the amount implied by the SoC
  estimate,
- voltage recovery and cutoff behavior can dominate the observed terminal
  voltage,
- the current SoC estimate must be cross-checked with voltage behavior.

For future calibration, do a controlled rest-voltage test instead of deriving an
OCV curve from deployment logs:

Recommended procedure:

1. Move the battery to a target SoC, such as 20%, 40%, 60%, or 80%.
2. Stop both charging and discharging.
3. Let the battery rest while keeping the load and motor off.
4. Record battery voltage and current after 0, 1, 5, and 10 minutes of rest.
5. If time allows, repeat the same target SoC twice:
   once after charging down/up to that point, and once after discharging to that
   point. The two voltages may differ because of voltage relaxation and
   hysteresis.

The goal is to measure the voltage that the battery naturally settles to at each
SoC, instead of using a voltage observed immediately during charge/discharge.

## SoH Integration Status

The repository includes:

- `core/soh_predictor/online.py` for online SoH prediction orchestration,
- GUI fields for selecting SoH model artifacts,
- deployment logging fields for SoH prediction status,
- a release `soh_models/` folder.

However, high-accuracy real-time SoH prediction for this experimental battery is
not currently validated. Treat SoH support as an integration interface and data
collection path. Do not rely on SoH-adjusted capacity in hardware deployment
unless the selected SoH model has been validated with battery-specific aging
data.

## Flow-Rate Integration Status

The codebase supports flow-rate action interfaces and a simulation-side
equivalent model. On the current small P302 platform, flow-rate economic
optimization should be deferred because motor power can dominate the energy
balance.

Recommended current position:

- Keep GUI / command / logging support for flow rate.
- Use fixed flow during hardware deployment.
- Record measured `flow_percent -> motor_power_w`.
- In future simulation updates, model motor power as:

```text
total_demand = external_load + motor_power(flow)
```

instead of treating it only as a small pump loss.

## What Is Safe to Change

Low-risk changes:

- Add diagnostics under `data/scripts/diagnostics/`.
- Add report figures under `data/scripts/figures/` without committing generated
  images by default.
- Add new configs under `configs/experiments/p302/`.
- Improve docs under `docs/`.
- Add tests for parser, deployment guards, and environment behavior.

Higher-risk changes:

- Changing `control/io_protocol.py` field parsing.
- Changing `control/run_deployment.py` guard order.
- Changing current sign conventions.
- Changing `SoCTracker` capacity or efficiency semantics.
- Enabling `--soh-use-for-capacity` on hardware without validation.
- Treating flow rate as an RL action on hardware without motor-power budgeting.

## Do Not Commit

Keep these local or external:

- raw deployment CSV logs,
- generated figures and PDFs,
- packaged release folders,
- local `config_gui.json`,
- large RL checkpoints,
- vendor controller folders,
- machine-specific paths.

See `docs/repository_hygiene.md` for the public repository policy.

## Useful Validation Commands

Run these before committing code changes:

```powershell
py -m compileall core control data\scripts tests gui
py -m pytest
git status --short
```

Before pushing, check for accidental local paths or generated artifacts:

```powershell
rg "C:\\Users|Data_ID0|P302_AI_V4.0|Microgrid_AI" .
git status --short --untracked-files=all
```

## Recommended Next Steps

For a future student or maintainer:

1. Keep hardware deployment focused on safe charge/discharge control first.
2. Collect controlled motor-power data for each flow setting.
3. Keep flow fixed in hardware until the battery module can supply both load and
   motor power with a meaningful margin.
4. Collect controlled SoC/rest-voltage data before using voltage as an OCV curve.
5. Validate any SoH model on battery-specific aging data before using it for
   capacity correction.
6. Preserve the separation between raw policy action, safety-corrected action,
   and final hardware command in all logs.

## Research Framing

The strongest research contribution is not that every battery-health feature is
fully solved. The contribution is that the project exposes and handles the
real-world deployment gap:

- hardware action semantics differ from simulation,
- SoC is an operational estimate rather than a direct state measurement,
- voltage cutoff and current-sign behavior must be guarded,
- auxiliary loads can dominate a small battery platform,
- safety correction must be logged and analyzed separately from raw policy
  intent.

This framing is important for continuing the thesis, writing the final report,
and explaining why some interfaces are implemented but not fully enabled on the
current hardware scale.

## Thesis Direction Update

Latest thesis decision from 2026-06-22:

- Keep the main theme as a safety- and profit-oriented microgrid energy
  management system.
- Move the narrative order earlier toward physical-platform alignment: first
  establish a simulation environment that matches the real P302 platform, then
  reserve a simulation-vs-hardware validation section, and only afterward use
  the trusted simulation to compare control strategies.
- Do not fill substantive simulation-vs-hardware validation results yet. Keep
  only the section structure, validation plan, and placeholder until enough real
  data is available.
- Put complex but necessary implementation details in the appendix rather than
  overloading the main text.
- For `thesis_sim`, first try longer validation/evaluation windows to see
  whether safety, profit, greedy baselines, and RL/CORAL results separate more
  clearly. Add export / sell-back simulation only as a later option if the
  longer validation still cannot support a clear thesis story.
- The main heuristic baseline family is safety-first, profit-first, and
  balanced safety-profit greedy. Do not present a perfect expert rule as the
  primary baseline.
