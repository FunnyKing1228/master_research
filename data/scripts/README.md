# Data Scripts

Reusable data and analysis scripts are grouped by workflow stage.

```text
data/scripts/
  preprocessing/   Convert raw logs into training/evaluation datasets
  diagnostics/     Inspect deployment anomalies, sensor freezes, and PV/load gaps
  figures/         Generate report-ready analysis figures
  baselines/       Build and summarize baseline comparison runs
  archive/         Older one-off scripts kept only for traceability
```

These scripts may read raw deployment CSVs from `data/raw/`, but raw logs and
generated figures are intentionally ignored by default. Commit scripts and small
configuration files; keep generated data, plots, and local reports outside the
public branch unless they are explicitly referenced by documentation.
