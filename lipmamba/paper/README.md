# Paper artefacts

This folder holds the machine-rendered LaTeX replacement text for the red
`\todo` markers of `lipmamba_iclr2027.tex`.

```
python scripts/regenerate_all.py --config configs/lipmamba_130m.yaml \
    --checkpoint runs/lipmamba_130m/final.pt --tokens data_cache/wikitext103_val.bin
# → runs/todo{1,2,4,5}_*.json, runs/*.png, paper/todo_snippets.tex
```

`todo_snippets.tex` contains, for each marker, a paragraph with the measured
numbers substituted, plus pgfplots coordinates for Figures 2 and 4.  Paste
them over the markers; nothing is generated for a marker whose experiment has
not been run (it is listed under "STILL OPEN" instead).

The manuscript sources themselves are **not** stored here: the journal
versions live in <https://github.com/rogerpanel/LipMamba-Models>, and the
ICLR version must be kept out of any public repository until the
double-blind period ends.

See `docs/ICLR2027_AUDIT_RESPONSE.md` for the mapping from audit findings to
code and scripts, and for the two numerical facts (ℓ\* ≈ 1; Theorem-1
constant ≈ 10⁸ per block) that the text must be made consistent with.
