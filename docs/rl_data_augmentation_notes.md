# RL Data Augmentation Notes

This note records candidate data augmentation and robustness-validation ideas for the microgrid RL policy. These items are planned directions, not completed experiments unless later results are added.

## Candidate Augmentations

- **PV scaling**: multiply the PV profile by factors such as `0.6`, `0.8`, `1.0`, and `1.2` to simulate cloudy, normal, and sunny conditions.
- **Measurement noise**: add small random noise to PV, load, current, voltage, or SoC observations to test whether the policy is robust to sensor uncertainty.
- **Time shifting**: shift PV/load profiles by 15, 30, or 60 minutes to reduce dependence on fixed time-index patterns.
- **Initial SoC randomization**: evaluate or train episodes with different starting SoC values, such as 30%, 50%, and 70%.

## Thesis Framing

These augmentations are useful because a repeated historical dataset can make a policy appear strong even if it partially memorizes time-indexed patterns. Perturbing PV, load, timing, noise, and initial SoC helps evaluate whether the policy learned a state-dependent control strategy rather than a fixed action schedule.

## Suggested Minimal Validation

For a lightweight robustness test, evaluate the trained CORAL policy without retraining under:

- PV scaling: `0.8x`, `1.0x`, `1.2x`
- Initial SoC: `30%`, `50%`, `70%`
- Optional load scaling: `0.8x`, `1.0x`, `1.2x`

Report profit, SoC violation count, realized safety violations, and qualitative charge/discharge timing.
