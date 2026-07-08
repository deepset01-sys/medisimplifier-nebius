# Judge Calibration Plan — Perturbation-Based
Source: Fable 5 consultation, July 2026

## Architecture
- Perturb REFERENCES (Claude Opus 4.5 outputs), not model predictions
- n=150 per error type + 200 clean controls = ~800 Token Factory calls per judge = 1,600 total
- Run only on samples where judge PASSED the unperturbed version

## Error Types
1. perturb_dose — 10× medication order-of-magnitude (regex)
2. perturb_negation — 8 negation-flip patterns
3. perturb_drop_diagnosis — remove sentence with diagnosis cues
4. perturb_lateral — swap left↔right (anatomical context only)

## Output Format
| Error type | n | Llama sens. (95% CI) | Qwen sens. | Either-judge | Mean ΔROUGE-L |
|------------|---|----------------------|------------|--------------|---------------|

Plus specificity row on 200 clean controls.

## Schedule
- Day 1: perturbation_calibration.py + QC → perturbed_calibration_set.json
- Day 2: Token Factory runs → raw verdicts
- Day 3: sensitivity/specificity + Wilson CIs + README update

## Cost
~$5-10 on Token Factory
