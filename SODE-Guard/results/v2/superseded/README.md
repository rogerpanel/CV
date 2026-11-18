# Superseded results: do not use

`runs_iis3d_cicids2018_file_size_leak/` and `summary_iis3d_cicids2018_file_size_leak.md`
come from the first CSE-CIC-IDS2018 (IIS3D) run. That run used `file_size_mb`, an integration
metadata column with 10 distinct values (one per daily capture file), as a feature. Because
each capture day contains specific attacks, the column encodes the label, so these results
are invalid. `prepare` now drops such columns, and the dataset is re-run from scratch.
