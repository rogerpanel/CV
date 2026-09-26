"""Write paper/integrated/e7_paragraph.tex from results/real/E7/e7_permutation.csv."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
t = pd.read_csv(ROOT / "results/real/E7/e7_permutation.csv")
g = t.groupby(["cohort", "model", "splitter"]).agg(w=("window_auroc_est", "mean"), wmin=("window_auroc_est", "min"), wmax=("window_auroc_est", "max"), s=("subject_auroc_est", "mean")).reset_index()

def r(c, m, sp, col):
    return float(g[(g.cohort == c) & (g.model == m) & (g.splitter == sp)][col].iloc[0])

parts = []
for c in ("depresjon", "psykose", "hyperaktiv"):
    parts.append(
        f"On {c.upper()}, subject-wise AUROC on permuted labels was {r(c,'logreg','subject_wise','w'):.3f} (logistic regression) and {r(c,'xgboost','subject_wise','w'):.3f} (XGBoost), "
        f"whereas record-wise AUROC was {r(c,'logreg','record_wise','w'):.3f} (range over permutations {r(c,'logreg','record_wise','wmin'):.3f} to {r(c,'logreg','record_wise','wmax'):.3f}) and "
        f"{r(c,'xgboost','record_wise','w'):.3f} ({r(c,'xgboost','record_wise','wmin'):.3f} to {r(c,'xgboost','record_wise','wmax'):.3f}), reaching {r(c,'logreg','record_wise','s'):.3f} and {r(c,'xgboost','record_wise','s'):.3f} at participant level."
    )
n_perm = int(t.permutation.nunique())
text = (f"With diagnostic labels randomly reassigned across participants ({n_perm} permutations; each participant keeps one label, but which one is random), "
        "no disorder signal remains and an honest estimate must sit at 0.5 (Table~\\ref{tab:e7}). " + " ".join(parts) +
        " Under record-wise splitting the models therefore achieved discrimination comparable to the \\emph{genuine} subject-wise results of E1 on labels that carry no information about the disorder at all: "
        "the design measures the model's ability to recognise a participant from other days of the same participant, and nothing else. "
        "The effect was larger for XGBoost than for logistic regression in every cohort, consistent with the E1 ordering.\n")
(ROOT / "paper/integrated/e7_paragraph.tex").write_text(text)
print(text)
