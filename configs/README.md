# Configurations

This directory separates reusable configuration files by purpose.

```text
configs/
  config_p302_sim.yaml        Default quick-start simulation config
  config_gui.template.json    Public GUI configuration template
  deployment/                 Reserved for hardware deployment presets
  experiments/p302/           Historical and reproducible P302 RL experiments
  baselines/research/         Baseline and ablation configs used for comparisons
```

Raw experiment outputs, local machine paths, and generated command manifests
should stay out of Git. If a config depends on private data, keep the relative
path format and document the expected dataset separately.
