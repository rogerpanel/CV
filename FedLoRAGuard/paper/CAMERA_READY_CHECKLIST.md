# Camera-ready checklist — EMNLP 2026 Findings

Deadline: **30 August 2026, Anywhere-on-Earth (AoE)**. The camera-ready
deadline is final; extensions are not possible.

The items below map exactly to the EMNLP 2026 Program Chairs'
camera-ready notification. Tick each one before pressing "Submit" on
OpenReview.

---

## 1. Reference integrity — CRITICAL

Program Chairs' quote: *"Papers flagged for irresponsible AI use during
this screening may be subject to additional review. If a camera-ready
paper is determined to violate these policies, such as containing
hallucinated, fabricated, or non-existent references, the paper will be
desk rejected from the conference."*

The camera-ready cites **57 works**. All are inherited from the accepted
review-version manuscript (v6). Do **NOT** rely on AI to verify them —
open each one manually.

### Verify by opening each URL

| Reference key | What to verify | Where |
| --- | --- | --- |
| `liu2024loraattack` | `arXiv:2403.00108` resolves and title matches "LoRA-as-an-Attack" | https://arxiv.org/abs/2403.00108 |
| `lermen2023lora` | `arXiv:2310.20624` resolves | https://arxiv.org/abs/2310.20624 |
| `schulz2025shadowgenes` | `arXiv:2502.04321` — v6 review draft had `2501.11830`; confirm which arXiv ID is real and correct | https://arxiv.org/abs/2502.04321 (and cross-check 2501.11830) |
| `kellas2025pickleball` | `arXiv:2508.15987`, CCS 2025 | https://arxiv.org/abs/2508.15987 |
| `mironov2019sampled` | `arXiv:1908.10530` | https://arxiv.org/abs/1908.10530 |
| `fan2023fatellm` | `arXiv:2310.10049` | https://arxiv.org/abs/2310.10049 |
| `ye2024openfedllm` | `arXiv:2402.06954`, KDD 2024 | https://arxiv.org/abs/2402.06954 |
| `tao2026safefedllm` | `arXiv:2601.07177`, ACL 2026 — the arXiv ID `2601.xxxxx` implies January 2026; if this paper does not yet exist on arXiv, replace with the exact preprint identifier the ACL 2026 acceptance carries | https://arxiv.org/abs/2601.07177 |
| `rahman2025hugginggraph` | `arXiv:2507.14240` | https://arxiv.org/abs/2507.14240 |
| `horwitz2025atlas` | `arXiv:2503.10633` | https://arxiv.org/abs/2503.10633 |
| `beutel2020flower` | `arXiv:2007.14390` | https://arxiv.org/abs/2007.14390 |

### Verify by opening each DOI (author self-citations)

| Reference key | DOI | Confirm |
| --- | --- | --- |
| `anaedevha2026byzantine` | `10.1007/s40747-026-02412-2` (Complex & Intelligent Systems, in press) | https://doi.org/10.1007/s40747-026-02412-2 |
| `anaedevha2026multiscale` | `10.1109/EDM69524.2026.11632205` (IEEE EDM 2026) | https://doi.org/10.1109/EDM69524.2026.11632205 |
| `anaedevha2026mambashield` | `10.1016/j.eswa.2026.131175` (Expert Systems with Applications) | https://doi.org/10.1016/j.eswa.2026.131175 |
| `anaedevha2026gp` | `10.1016/j.neucom.2026.133105` (Neurocomputing 677:133105) | https://doi.org/10.1016/j.neucom.2026.133105 |

**Advisory**: Springer, Elsevier and IEEE DOIs typically encode the
publication year in a specific slot; some of the DOIs above use `026-`
or `.2026.` which are not the usual house style — please open each URL
and confirm it lands on the real published article.

### Verify by name-check (well-known conference papers)

The remaining 42 items point to standard, well-indexed conference
publications (ICLR / NeurIPS / ICML / NAACL / KDD / WWW / S&P / CCS /
NDSS / MLSys / AISTATS / IEEE Access / CIKM / USENIX). For each one,
open the venue proceedings page for the stated year and confirm the
authors and title are correct. A grep of exact citation keys is in the
next section.

Full list to fact-check (title match + author list match + year match):

```
hu2021lora, dettmers2023qlora, zhang2023adalora, liu2024dora,
zhao2024galore, huang2024composite, wei2024brittleness, sun2025peftguard,
cohen2024jfrog, kurita2020weight, li2025backdoorllm, mcmahan2017fedavg,
li2020fedprox, karimireddy2020scaffold, abadi2016deep, mironov2017renyi,
dong2022gaussian, gopi2021numerical, bonawitz2017practical,
blanchard2017machine, yin2018byzantine, cao2021fltrust, wang2024flora,
kumar2019jodie, trivedi2019dyrep, xu2020tgat, rossi2020tgn,
yu2023dygformer, hu2020hgt, he2021fedgraphnn, sajadmanesh2023gap,
lecuyer2019certified, cohen2019certified, ma2019data, xie2021crfl,
cao2022flcert, wang2021certgnn, scholten2022randomized,
loshchilov2019adamw, sharafaldin2018toward, ferrag2022edge,
moustafa2015unsw, moustafa2021ton
```

---

## 2. Page-limit compliance

EMNLP 2026 Findings, long paper, camera-ready allowance:

* Content (§1 through Conclusion): **up to 9 pages**
* Limitations (required): **+1 page** (up to 1 page)
* Ethics Statement (optional): +1 page allowed
* References: **unlimited**
* Appendices / supplementary: **unlimited**

The main body of `FedLoRAGuard_camera_final.tex` has been **condensed**
to bring the page count within budget while preserving every reviewer-
promised revision.  Pooled stats:

* Old main-body (§1–Ethics): 1191 lines of TeX source
* New main-body: 853 lines (**−28%**)
* Stub-compile page count went from 22 → 18 pages (single-column A4)

In the real Overleaf compile with `acl.sty` (two-column Letter, ~1.5×
denser packing), the condensed main body is expected to occupy roughly
**6–7 pages for §1–§Conclusion + 1 page Limitations + ~½ page Ethics**,
well within the 11-page pre-References budget.

Verify on Overleaf:

* Compile with the real EMNLP 2026 `acl.sty` (two-column Letter).
* Open the PDF and confirm References begins on **page 12 or earlier**
  (§1–§Conclusion ≤ 9 pages, Limitations ≤ 1 page, Ethics ≤ 1 page).
* If the compile is close to the limit, prime candidates for further
  trimming without losing reviewer-promised content are: the
  "Numerical instantiation" paragraph after Theorem 2, one of the two
  centralised-baseline lines in Table 1, and the "Additional analyses"
  overview sentence in §6.4.
* If it exceeds by more than half a page, revert to the two-paragraph
  form of the Introduction "Our approach" section by re-adding the
  "Claim structure" paragraph from `git log -p paper/FedLoRAGuard_camera_final.tex`
  (commit before this condensation).

Both figures (`fig1_FedLoraGuard_arch.pdf`, `fig2_mean.pdf`) declare
`figure*[t]`. Confirm on Overleaf that they land near their reference
and are not pushed to the last page; if they overflow, change to
`figure*[!ht]` or use single-column `figure[t]`.

---

## 3. Responsible NLP Checklist — update on OpenReview

The Program Chairs require the Responsible NLP Checklist to be
**re-filled** on the OpenReview commitment page. Section numbers you
referenced in the review-version checklist and their status in the
camera-ready:

| Checklist item | Review-version section | Camera-ready section | Change? |
| --- | --- | --- | --- |
| B1 Cite creators of artifacts | §§1, 5 | §§1, 6 | Renumbered (§5 became §6 Experiments) |
| B2 Discuss license | Ethics + App. B | Ethics + App. E (Reproducibility) | Appendix letter changed |
| B3 Intended-use consistency | Ethics | Ethics | Same |
| B4 PII / offensive content | Ethics | Ethics | Same |
| B5 Documentation of artifacts | §5 + App. B | §6 + App. E | Renumbered |
| B6 Statistics | §5 + App. B | §6 + App. E | Renumbered |
| C1 Model size / budget | App. B | App. E | Renumbered |
| C2 Setup / hyperparameters | §§5, 6, App. B | §6 + App. E | Renumbered |
| C3 Descriptive statistics | §6 | §6 | Same |
| C4 Package parameters | App. B | App. E | Renumbered |
| E1 AI assistants | Ethics | Ethics | Same |

Update every section-number reference in the Responsible NLP form on
OpenReview to match the camera-ready. The camera-ready main-body
section order is unchanged, only Appendices B → E (I've renamed the
Reproducibility appendix from `B` to `E` to fit new appendices A/B/C/D
for proofs / analyses / IDS / provenance).

---

## 4. Author, presenter and travel information (OpenReview form)

The Program Chairs need the following data on the OpenReview
camera-ready page. Fill in exactly as the notification lists.

* Paper Registration
  * Registrar's name and email
  * Presenting-author name and email
* Presenting-author country of residence
* Visa requirement (Yes/No)
  * If Yes: (i) already have visa, (ii) applied, or (iii) need to apply
* Invitation letter needed? (form
  https://forms.gle/6u3TuNotWMALBRm7A)
* Presentation (In-person / Virtual) and preferred mode (Oral / Poster)
* If in-person: anticipated travel dates

Camera-ready author metadata already in the `.tex`:

```
Roger Nick Anaedevha (corresponding: ar006@campus.mephi.ru)   NRNU MEPhI
Alexander G. Trofimov                                          NRNU MEPhI
Yuri V. Borodachev                                             NRNU MEPhI (AI Research Center)
```

Confirm the presenting author, ORCID iDs on OpenReview, and that the
email `ar006@campus.mephi.ru` is monitored through the deadline.

---

## 5. Anonymity, integrity, and consistency

Sanity checks the Program Chairs will apply automatically:

* [ ] No `\anonymous` / `\review` toggles remain — my camera-ready uses
      `\usepackage[final]{acl}` and no "anonymous version" comments.
* [ ] Author names, affiliations, and acknowledgements are present.
* [ ] AI-assistant disclosure paragraph is consistent with the E1
      OpenReview form (**already present** in Ethics Statement).
* [ ] Every figure and table is referenced from the text.
* [ ] Every `\cite{...}` key resolves to a `\bibitem`. Automated cross-
      check already passes (57 keys → 57 bibitems + 1 unused
      `kurita2020weight`).
* [ ] Compile with `pdfLaTeX` (not XeLaTeX), two passes.

---

## 6. Files to upload to OpenReview

* `FedLoRAGuard_camera_final.tex` (or renamed as your submission
  template requires)
* `figures/fig1_FedLoraGuard_arch.pdf`
* `figures/fig2_mean.pdf`
* Signed EMNLP author agreement (via OpenReview form)
* Updated Responsible NLP Checklist (via OpenReview form)

---

## 7. Post-submission (from the notification)

* Program Chairs finalise the schedule ~2 weeks after 30 Aug AoE.
* Confirm registration for the paper (at least one author).
* Visa Invitation Letter Request form:
  https://forms.gle/6u3TuNotWMALBRm7A
* Program-Chair contact: emnlp2026-programchairs@googlegroups.com
