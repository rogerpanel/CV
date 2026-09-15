"""Write paper/integrated/e6_paragraph.tex from results/real/E6/e6_ablation.csv (no hand-typed numbers)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
t = pd.read_csv(ROOT / "results/real/E6/e6_ablation.csv")

def rng(df, col):
    return f"{df[col].min():.3f} to {df[col].max():.3f}"

parts = []
for coh in ("depresjon", "psykose", "hyperaktiv"):
    g = t[t.cohort == coh]
    base = g[g.variant == "all_features"]
    drop = g[g.variant.str.startswith("drop_")]
    only = g[g.variant.str.startswith("only_")]
    smote = g[g.variant == "all_features+smote_in_fold"]
    loso = g[g.variant == "all_features+loso"]
    worst_only = only.loc[only.window_auroc_est.idxmin()]
    best_only = only.loc[only.window_auroc_est.idxmax()]
    parts.append(
        f"On {coh.upper()}, the full feature set gave AUROC {rng(base, 'window_auroc_est')} (logistic regression, XGBoost); "
        f"dropping any single feature group changed it to {rng(drop, 'window_auroc_est')}; single groups alone ranged from "
        f"{worst_only.window_auroc_est:.3f} ({worst_only.variant.replace('only_', '').replace('_', '/')}, {worst_only.model}) to "
        f"{best_only.window_auroc_est:.3f} ({best_only.variant.replace('only_', '').replace('_', '/')}, {best_only.model}); "
        f"SMOTE inside training folds gave {rng(smote, 'window_auroc_est')} with ECE {rng(smote, 'window_ece_est')}; "
        f"leave-one-subject-out gave {rng(loso, 'window_auroc_est')}."
    )
text = ("Honest performance was robust to the analytic choices a developer could plausibly vary (Table~\\ref{tab:e6}). "
        + " ".join(parts)
        + " No variant on HYPERAKTIV rose above chance under subject-wise evaluation, and no variant on DEPRESJON or PSYKOSE "
        "approached the record-wise estimates of E1: the gap between published and honest numbers is a property of the "
        "splitting design, not of feature engineering, resampling or the cross-validation scheme.\n")
(ROOT / "paper/integrated/e6_paragraph.tex").write_text(text)
print(text)
