# Kaggle results drop-off

Upload `results.zip` from the Kaggle notebook's Output tab here
(GitHub: *Add file* → *Upload files*, on branch
`claude/implement-mambaguard-detection-zCgu6`). Then summarise it with

```bash
python scripts/summarise_kaggle_results.py kaggle/results/results.zip --latex \
    --json reproducibility/kaggle_results.json
```

Do not upload `results_checkpoints.zip` (model weights) or any benchmark data.
