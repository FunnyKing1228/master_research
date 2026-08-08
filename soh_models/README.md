Place SoH prediction model files here.

Suggested contents:
- Transformer model weights, for example `merged_slot_v8_transformer.pth`.
- Scaler/preprocessing files, for example `merged_slot_v8_transformer_scaler.pkl`
  and `scaler_params.npz`.

The GUI keeps this path in `config_gui.json` as `soh_model_path`.
The inference code is already included in the packaged UI, so this folder does
not need the standalone `SOH_Predictor` source code, examples, or training logs.

When online SoH prediction is enabled, deployment writes candidate segments to
`<log_dir>/soh_online_segments/cycles` and predictions to
`<log_dir>/soh_online_segments/soh_online_predictions.csv`.
