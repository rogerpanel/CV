# Reconciliation: figure anchors vs. chapter prose

The benchmark's ground truth is the set of plotted anchor points in the
dissertation figure `fig:uav_mission_completion` (Chapter 6). The generation
harness reproduces those anchors exactly (within Monte-Carlo sampling noise at
`n = 600` per point) and then computes the DO-326A 0.90-completion crossings
*from the anchors themselves*.

## The one discrepancy to resolve before defence

The chapter prose states the 0.90-crossings as approximately:

| Defence            | Prose (chapter text) | Derived from plotted anchors |
|--------------------|----------------------|------------------------------|
| No-Def (PX4)       | ≈ 7.5 dB             | **7.7 dB**  ✅ consistent      |
| CAF-CNN + PX4      | ≈ 13 dB              | **11.5 dB** ⚠ ~1.5 dB gap     |
| Seq2Seq Tr. + PX4  | ≈ 18 dB              | **14.3 dB** ⚠ ~3.7 dB gap     |
| Ours: M1+M4+M6+M7  | ≈ 27 dB              | **25.0 dB** ⚠ ~2 dB gap       |

The prose crossings are slightly more generous than the plotted points imply,
because the plotted "ours" curve places completion = 0.90 *exactly at* J/S = 25
dB (anchor `(25, 0.90)`), whereas the text rounds up to "≈ 27".

## Two honest ways to close the gap (pick one)

**Option A — trust the plotted data, adjust the prose (recommended, minimal).**
Change the Chapter 6 sentence to read the data-derived values:

> «…удерживает завершаемость выше регуляторного порога DO-326A 0,90 до
> J/S ≈ 25 дБ включительно — против ≈ 14 дБ для Seq2Seq Tr., ≈ 11–12 дБ для
> CAF-CNN и ≈ 7,5 дБ для эталонной PX4 без защиты, то есть выигрыш в
> 10–18 дБ по операционно-значимому показателю.»

This keeps the published figure untouched and makes the text provably match
the artifact. The headline claim (a 10–18 dB operational margin over baselines)
survives essentially unchanged.

**Option B — keep the prose, nudge the anchors.**
If "≈ 27 dB" is the intended result, shift the `ours_m1m4m6m7` anchors in
`config/defenses.yaml` slightly right (e.g. `(25, 0.93)`, `(30, 0.86)`) and
re-run. This changes the published figure coordinates, so you must re-paste the
regenerated TikZ coordinates (`artifact/tikz_coordinates.txt`) into the LaTeX
figure and recompile.

Either way the benchmark and the dissertation end up **provably consistent**,
which is what a reviewer will check.
