"""Graphical abstract for the J-BHI submission.

Three panels read left to right, all from the generated tables in paper/empirical/tables
(E1 window-level AUROC, E7 label permutation, E8 participant re-identification). The
values are hard-coded from those tables so the figure is reproducible without the
raw data; re-run scripts/part2_extra_figures.py first if the tables change.

Output: paper/jbhi/graphical_abstract.{png,pdf} (landscape, 300 dpi) and
paper/jbhi/graphical_abstract_text.txt (the <=50-word table-of-contents caption).
"""
from __future__ import annotations

import pathlib
from decimal import Decimal, ROUND_HALF_UP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

OUT = pathlib.Path(__file__).resolve().parents[1] / "paper" / "jbhi"

# Palette (validated: series 1 blue, series 2 orange; neutral for chance/reference)
SUBJ, REC, NEUTRAL = "#2a78d6", "#eb6834", "#8a8985"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#ffffff"

COHORTS = ["DEPRESJON\n(depression)", "PSYKOSE\n(schizophrenia)", "HYPERAKTIV\n(ADHD)"]

# E1: window-level AUROC, XGBoost, seed-averaged (Table e1_auroc_inflation)
E1_SUBJ = [0.764, 0.903, 0.442]
E1_REC = [0.878, 0.945, 0.593]
# E7: AUROC on permuted labels, XGBoost, mean over permutations (Table e7_permutation)
E7_SUBJ = [0.517, 0.506, 0.533]
E7_REC = [0.744, 0.733, 0.671]
# E8: top-1 participant re-identification from one day, random forest (Table e8_reidentification);
# HYPERAKTIV 0.265 is printed as 27% (round half up), matching the 27 to 40% range in the text.
E8_TOP1 = [0.388, 0.397, 0.265]
E8_CHANCE = [0.018, 0.019, 0.013]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#d8d7d2",
    "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def style(ax, ylabel, ymax=1.0):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.yaxis.grid(True, color="#ecebe6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xticks(range(3), COHORTS, fontsize=9.5, color=INK)


def paired_bars(ax, a, b, label_a, label_b, fmt="{:.2f}"):
    w = 0.34
    xs = range(3)
    ba = ax.bar([x - w / 2 - 0.02 for x in xs], a, w, color=SUBJ, label=label_a, zorder=3)
    bb = ax.bar([x + w / 2 + 0.02 for x in xs], b, w, color=REC, label=label_b, zorder=3)
    for bars in (ba, bb):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012, fmt.format(r.get_height()),
                    ha="center", va="bottom", fontsize=9, color=INK)
    return ba, bb


fig = plt.figure(figsize=(13, 6.6), dpi=300)
gs = fig.add_gridspec(2, 3, height_ratios=[0.16, 1], hspace=0.05, wspace=0.34,
                      left=0.055, right=0.985, top=0.83, bottom=0.2)

# Headline
fig.text(0.5, 0.955, "Day-level evaluation of actigraphy models scores participant identity, not illness",
         ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
fig.text(0.5, 0.905, "Same features, same models, four public cohorts (DEPRESJON, PSYKOSE, HYPERAKTIV, OBF-Psychiatric); "
         "only the unit of cross-validation differs",
         ha="center", va="center", fontsize=10.5, color=INK2)

# Panel A: E1 inflation
axA = fig.add_subplot(gs[1, 0])
paired_bars(axA, E1_SUBJ, E1_REC, "Subject-wise split", "Day-level split")
style(axA, "Window-level AUROC (XGBoost)")
axA.axhline(0.5, color=NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
axA.set_title("A  Day-level splitting inflates AUROC\n     by 0.04 to 0.15", loc="left", fontsize=11.5, color=INK, fontweight="bold")

# Panel B: E7 permutation
axB = fig.add_subplot(gs[1, 1])
paired_bars(axB, E7_SUBJ, E7_REC, "Subject-wise split", "Day-level split")
style(axB, "AUROC on randomly permuted labels")
axB.axhline(0.5, color=NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
axB.set_title("B  With labels shuffled, day-level models\n     still reach AUROC 0.67 to 0.74", loc="left",
              fontsize=11.5, color=INK, fontweight="bold")

# Panel C: E8 re-identification
axC = fig.add_subplot(gs[1, 2])
pct = lambda v: str(Decimal(v * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)) + "%"
bt = axC.bar(range(3), E8_TOP1, 0.5, color=REC, zorder=3)
for r, v in zip(bt, E8_TOP1):
    axC.text(r.get_x() + r.get_width() / 2, v + 0.008, pct(v), ha="center", va="bottom", fontsize=9, color=INK)
style(axC, "Top-1 participant re-identification", ymax=0.5)
axC.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], ["0%", "10%", "20%", "30%", "40%", "50%"])
axC.axhline(max(E8_CHANCE), color=NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
axC.text(-0.45, 0.028, "chance 1 to 2%", fontsize=8.5, color=INK2, ha="left", va="bottom")
axC.set_title("C  One day of activity re-identifies its\n     participant 27 to 40% of the time", loc="left",
              fontsize=11.5, color=INK, fontweight="bold")

# One legend row for all panels
h, l = axA.get_legend_handles_labels()
h.append(Line2D([0], [0], color=NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)))); l.append("Chance level")
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=3, frameon=False, fontsize=10)

# Bottom takeaway band
band = FancyBboxPatch((0.055, 0.010), 0.93, 0.062, boxstyle="round,pad=0.004,rounding_size=0.008",
                      transform=fig.transFigure, facecolor="#eef3fb", edgecolor="none")
fig.patches.append(band)
fig.text(0.52, 0.041, "Leakage-safe protocol: subject-wise splits, frozen cross-cohort transfer, calibration, permutation and "
         "re-identification checks.  Of 28 published models on these cohorts, 21 split by day or did not say;  "
         "every accuracy above the honest band coincides with day-level splitting.",
         ha="center", va="center", fontsize=9.6, color=INK, wrap=True)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "graphical_abstract.png", dpi=300, facecolor=SURFACE)
fig.savefig(OUT / "graphical_abstract.pdf", facecolor=SURFACE)

TEXT = ("Actigraphy models for depression, schizophrenia and ADHD lose 0.04 to 0.15 AUROC when evaluated "
        "subject-wise instead of day-wise. Day-wise models score above chance on shuffled labels and re-identify "
        "participants from one day at 27 to 40 percent, showing they learn identity; published accuracies above the "
        "honest band used day-level splits.")
(OUT / "graphical_abstract_text.txt").write_text(TEXT + "\n")
print("words:", len(TEXT.split()))
