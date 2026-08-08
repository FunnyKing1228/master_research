# Carbon-Focused Livestock Microgrid SAC

This folder contains a carbon-oriented variant of the livestock farm microgrid
environment.

## Purpose

The earlier SAC policy learned mostly electricity price arbitrage. It reduced
cost, but it could increase grid energy and therefore increase CO2 emissions.

This version changes the reward direction:

- primary objective: reduce grid-import CO2
- secondary objective: keep some electricity-cost awareness
- reward PV utilization
- keep SoC and infeasible-discharge safety penalties

## Files

- `carbon_microgrid_env.py`: carbon-focused environment subclass
- `train_sac_carbon.py`: SAC training + 30-day rollout + daily/weekly impact figures

## Run

From the repository root:

```powershell
py -3 examples\livestock_farm_microgrid\carbon_focused\train_sac_carbon.py --total-timesteps 20000
```

Outputs are written to:

```text
examples/livestock_farm_microgrid/carbon_focused/results/
```
