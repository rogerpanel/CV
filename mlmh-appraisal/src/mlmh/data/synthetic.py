"""Synthetic actigraphy fixtures that mimic the on-disk layout of each dataset.

They exist so that the whole pipeline (loaders -> windows -> features -> CV ->
tables) can be exercised and tested without the real data, which cannot be
redistributed.  Every number produced from them is labelled SYNTHETIC in the
manifests and in the generated tables; none of it is a scientific result.

The generator is not a null model: cases have a flatter circadian rhythm and
more night-time activity than controls (a moderate, plausible effect), and each
participant carries an idiosyncratic "fingerprint" (mean level, phase, hourly
profile).  The fingerprint is what a record-wise split lets a model exploit, so
E1 on the fixture shows the *mechanism* of leakage inflation, not its size on
real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MIN = 1440


def _subject_series(rng: np.random.Generator, n_days: int, case: bool, start: pd.Timestamp) -> pd.DataFrame:
    # Population effects
    amp = rng.normal(1.0 if not case else 0.7, 0.15)
    night_level = rng.normal(0.08 if not case else 0.18, 0.04)
    mean_level = np.exp(rng.normal(np.log(260 if not case else 200), 0.25))
    # Participant fingerprint
    phase = rng.normal(0, 1.2)  # hours
    profile = rng.normal(0, 0.15, size=24)
    profile = np.repeat(profile, 60)
    t = np.arange(n_days * MIN)
    hour = (t % MIN) / 60.0
    circ = np.clip(np.sin(2 * np.pi * (hour - 6 - phase) / 24), 0, None) ** 1.5
    base = mean_level * (amp * circ + night_level) * (1 + np.tile(profile, n_days))
    noise = rng.gamma(shape=1.2, scale=1.0, size=t.size)
    act = np.clip(base * noise, 0, None)
    # sleep: zeros at night with probability
    asleep = (hour < 6.5 + 0.3 * phase) | (hour > 23.0 + 0.3 * phase)
    act = np.where(asleep & (rng.random(t.size) < (0.85 if not case else 0.7)), 0, act)
    # occasional missing minutes
    act = act.astype(float)
    act[rng.random(t.size) < 0.002] = np.nan
    ts = start + pd.to_timedelta(t, unit="min")
    return pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "date": ts.strftime("%Y-%m-%d"), "activity": pd.array(np.round(act), dtype="Int64")})


def _write(df: pd.DataFrame, path: Path, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep)


def generate(root: Path, seed: int = 0, n_days: int = 8, scale: float = 1.0) -> dict[str, Path]:
    """Create depresjon/, psykose/, hyperaktiv/ and obf_psychiatric/ under root."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2003-05-07 00:00:00")
    out = {}

    # DEPRESJON: 23 condition + 32 control (+ scores.csv)
    dep = root / "depresjon"
    n_cond, n_ctrl = max(3, int(23 * scale)), max(4, int(32 * scale))
    controls = []
    scores = []
    for i in range(1, n_cond + 1):
        s = _subject_series(rng, n_days, True, start)
        _write(s, dep / "condition" / f"condition_{i}.csv")
        scores.append({"number": f"condition_{i}", "days": n_days, "gender": int(rng.integers(1, 3)), "age": "35-39", "afftype": int(rng.integers(1, 4)), "madrs1": int(rng.integers(15, 30))})
    for i in range(1, n_ctrl + 1):
        s = _subject_series(rng, n_days, False, start)
        controls.append(s)
        _write(s, dep / "control" / f"control_{i}.csv")
        scores.append({"number": f"control_{i}", "days": n_days, "gender": int(rng.integers(1, 3)), "age": "35-39", "afftype": "", "madrs1": ""})
    _write(pd.DataFrame(scores), dep / "scores.csv")
    out["depresjon"] = dep

    # PSYKOSE: 22 patients + THE SAME 32 controls (this is how the real data is)
    psy = root / "psykose"
    for i in range(1, max(3, int(22 * scale)) + 1):
        s = _subject_series(rng, n_days, True, start)
        _write(s, psy / "patient" / f"patient_{i}.csv")
    for i, s in enumerate(controls, start=1):
        _write(s, psy / "control" / f"control_{i}.csv")
    out["psykose"] = psy

    # HYPERAKTIV: activity_data/patient_activity_XX.csv + patient_info.csv (semicolon separated)
    hyp = root / "hyperaktiv"
    info = []
    n_h = max(6, int(85 * scale))
    for pid in range(1, n_h + 1):
        adhd = int(rng.random() < 0.6)
        s = _subject_series(rng, max(2, n_days // 2), bool(adhd), start).rename(columns={"timestamp": "TIME", "activity": "ACTIVITY"})[["TIME", "ACTIVITY"]]
        _write(s, hyp / "activity_data" / f"patient_activity_{pid:02d}.csv", sep=";")
        info.append({"ID": pid, "SEX": int(rng.integers(0, 2)), "AGE": int(rng.integers(1, 5)), "ACC": 1, "ACC_TIME": "", "ACC_DAYS": n_days // 2, "HRV": 0, "ADHD": adhd, "ADD": 0, "BIPOLAR": 0, "UNIPOLAR": 0, "ANXIETY": 0})
    _write(pd.DataFrame(info), hyp / "patient_info.csv", sep=";")
    out["hyperaktiv"] = hyp

    # OBF-Psychiatric: five folders
    obf = root / "obf_psychiatric"
    for g, n, case in (("adhd", 6, True), ("clinical", 6, True), ("control", 8, False), ("depression", 6, True), ("schizophrenia", 6, True)):
        for i in range(1, max(3, int(n * scale * 2)) + 1):
            s = _subject_series(rng, n_days, case, start)
            _write(s, obf / g / f"{g}_{i}.csv")
    out["obf_psychiatric"] = obf
    return out
