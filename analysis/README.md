# Analysis

Use this directory for:

- trial logs
- success-rate tables
- ablation study plots
- failure case summaries

Suggested outputs:

- `results.csv`
- `success_rate_by_occlusion.png`
- `ablation_openvla_vs_refinement.png`

Quick summary:

```bash
python3 analysis/summarize_trials.py
```

This script reads `analysis/logs/trial_log.jsonl`, prints a compact summary, and writes:

- `analysis/logs/trial_summary.csv`
