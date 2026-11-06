# MambaGuard --- IEEE S&P 2027 Cycle 2 Submission Checklist

All times are AoE (Anywhere on Earth, UTC-12). Deadlines from the official
S&P 2027 Call for Papers, Cycle 2.

- Abstract registration: 2026-11-10 23:59 AoE
- Full submission:       2026-11-17 23:59 AoE

Each item is a checkbox (`[ ]`) followed by its acceptance criterion
(**AC:** ...). Do not tick an item until its AC holds.

---

## By 8 November 2026 --- preflight

### Concurrent-submission hygiene

- [ ] IEEE Access manuscript formally withdrawn.
      **AC:** Withdrawal email sent to the IEEE Access Editorial Office and
      an acknowledgement reply filed in `sp2027/withdrawals/ieee_access.eml`.
- [ ] TNNLS manuscript status resolved.
      **AC:** Either (a) withdrawal email + ack filed, or (b) written
      confirmation that the TNNLS submission is a strict subset covered by
      S&P's no-overlap exception, filed in `sp2027/withdrawals/tnnls.eml`.
- [ ] LipMamba-ICLR27 overlap resolved on paper and in audit.
      **AC:** `DUAL_SUBMISSION_AUDIT.md` row for LipMamba filled in; the
      S&P draft cites LipMamba in the third person and the overlapping
      Lipschitz-theory lemmas are flagged with a footnote stating prior
      publication status.

### Author list and metadata

- [ ] All co-author ORCIDs collected.
      **AC:** `sp2027/author_table.csv` has one row per author with columns
      `name, affiliation, email, orcid, is_student`, and every ORCID
      resolves at https://orcid.org/<id>.
- [ ] Final title locked.
      **AC:** Title string recorded in `sp2027/title.txt`, matches the
      manuscript `\title{...}`, and fits HotCRP's 200-char limit.
- [ ] Final abstract locked.
      **AC:** Abstract text in `sp2027/abstract.txt`, 250 +/- 50 words,
      no author names, no URL that deanonymizes.

### Infrastructure

- [ ] HotCRP account active with correct contact address.
      **AC:** Submitter has logged in at the S&P 2027 HotCRP site in the
      last 7 days.
- [ ] Six-paper-per-author cap check.
      **AC:** For every co-author, count of papers they appear on in S&P
      2027 (any cycle) across all submissions <= 6. Count written to
      `sp2027/paper_cap_check.txt`.

---

## By 10 November 23:59 AoE --- abstract registration

HotCRP fields to fill (exact names as they appear in the S&P 2027 form):

- [ ] Title.
      **AC:** Pasted from `sp2027/title.txt`, no trailing whitespace.
- [ ] Abstract.
      **AC:** Pasted from `sp2027/abstract.txt`, renders without LaTeX
      source in HotCRP preview.
- [ ] Authors.
      **AC:** Every co-author added with first name, last name,
      affiliation, email, ORCID. Order matches `sp2027/author_table.csv`.
- [ ] Conflicts (all 8 categories).
      **AC:** Full S&P 2027 PC list has been read line by line, and for
      each of the 8 CFP-defined conflict categories (same institution;
      co-author of a paper within 2 years; advisor/advisee ever; close
      personal relationship; direct financial relationship within 2 years;
      recent collaborator on grants; dispute; other listed category)
      a HotCRP checkbox has been reviewed and ticked where applicable.
      A paper copy of the flagged conflicts is saved to
      `sp2027/conflicts_declared.txt`.
- [ ] Topics selected.
      **AC:** At least 2 and at most 5 topics ticked, drawn from the CFP
      topic list; primary topic matches the paper's framing (ML security
      / LLM security / protocol security).
- [ ] Ethics Considerations statement.
      **AC:** Text from `ETHICS.md` pasted into the HotCRP field, length
      under HotCRP's limit. No author-identifying content.
- [ ] Generative AI usage statement.
      **AC:** Text from `GENAI_USAGE.md` pasted. Models named explicitly;
      editorial vs. research use distinguished.
- [ ] Six-paper-per-author cap re-checked after abstract list is final.
      **AC:** No change from the 8 Nov count; discrepancy, if any, resolved
      before submit.
- [ ] Abstract registered.
      **AC:** HotCRP page shows "Abstract registered" badge and the
      `sp2027/hotcrp_receipt_abstract.pdf` printout is saved.

---

## By 13 November 2026 --- content lock

Targets the S&P checklist of common revision reasons (C.1-C.12 are our
internal mapping of the CFP's "frequent reviewer concerns" guidance).

- [ ] C.1 Threat model unambiguous.
      **AC:** Section 2 defines attacker capability, knowledge, goal in
      three bulleted lines; no later section widens it silently.
- [ ] C.2 No overclaimed novelty.
      **AC:** Contributions list uses the words we can defend; each item
      cross-referenced to the subsection that evidences it.
- [ ] C.3 Baselines current.
      **AC:** At least one 2026 baseline per table; source code or weights
      traceable from the bibliography entry.
- [ ] C.4 Statistical significance reported.
      **AC:** Every headline number is mean +/- std over 5 seeds; CI or
      paired test for the key comparisons.
- [ ] C.5 Dataset provenance clear.
      **AC:** Section 4 lists each dataset with source, licence, and the
      synthetic/real split.
- [ ] C.6 Hyperparameter search reported, not hidden.
      **AC:** Appendix table of HP grids; our final HP is not the best of
      an unreported sweep on the test set.
- [ ] C.7 Compute footprint reported.
      **AC:** `COMPUTE_BUDGET.md` content folded into Section 5 and the
      Appendix; numbers consistent.
- [ ] C.8 Negative results not hidden.
      **AC:** The one dataset where MambaGuard underperforms is reported
      in-table, not only in the Appendix.
- [ ] C.9 Related work places us, not we place us.
      **AC:** Each related system described in its authors' terms before
      we contrast.
- [ ] C.10 Graph-edit baseline decision made and recorded.
      **AC:** Either (a) graph-edit attacker included and ablated, or
      (b) scope note in Section 2 explaining why it is out of scope for
      this paper. Decision recorded in `sp2027/graph_edit_decision.md`.
- [ ] C.12 Adaptive attacks executed.
      **AC:** At least one adaptive attack per defence component (PGD at
      the detector, GCG at the projector, LLM-driven rewriting, graph
      evasion) with results in Section 5.
- [ ] One protocol-native benchmark promoted to the main results table.
      **AC:** MCPSecBench OR MCP-SafetyBench OR A2ABreak sits in Table 1,
      not only an appendix.

---

## By 16 November 2026 --- production freeze (T-24h)

- [ ] IEEEtran compsoc port complete.
      **AC:** Document class line is
      `\documentclass[conference,compsoc,10pt]{IEEEtran}`; `make` builds
      with zero warnings except known-safe hyperref ones.
- [ ] 13-page body cap (excluding references and appendices).
      **AC:** `pdftotext` word count of pages 1-13 is the whole body;
      page 14 is the first references page.
- [ ] Figures and tables render in grayscale.
      **AC:** Preview PDF with
      `gs -dSAFER -sDEVICE=pdfwrite -sColorConversionStrategy=Gray`; no
      figure becomes unreadable.
- [ ] Anonymised artifact mirror frozen.
      **AC:** Run from the repo root as
      `sp2027/anonymize.sh . sp2027/` --- this excludes `.git/`,
      `manuscript_v4/` (the full-attribution LaTeX), and `sp2027/`
      (this submission package) from the mirror, redacts in-place, and
      must exit 0. SHA256 of the resulting tarball is recorded in
      `sp2027/anon_build.sha256`.
- [ ] Artifact mirror uploaded to anonymous.4open.science.
      **AC:** URL resolves, SHA256 of download matches `anon_build.sha256`,
      URL written into the paper's Artifact Availability appendix.
- [ ] `ARTIFACT_AVAILABILITY.md` content copied into the manuscript
      appendix.
      **AC:** URL in the paper is a live anonymous mirror, not a
      placeholder.
- [ ] HotCRP full submission uploaded >= 24 h before the deadline.
      **AC:** HotCRP shows upload timestamp <= 2026-11-16 23:59 AoE,
      receipt saved to `sp2027/hotcrp_receipt_paper.pdf`.
- [ ] Final anonymity scan on the PDF.
      **AC:** `pdftotext submission.pdf - | grep -E
      'Anaedevha|Trofimov|Borodachev|MEPhI|robustidps\.ai'` returns
      nothing. PDF metadata `Author` field is empty or "Anonymous".

---

## 17 November 2026 --- submission day

- [ ] Final HotCRP submit action pressed with at least 2 h margin.
      **AC:** Status badge "Submitted" visible; email receipt filed.
- [ ] Post-submit retrospective note opened.
      **AC:** `sp2027/postsubmit_notes.md` created with a one-line
      pointer to each open issue for rebuttal preparation.
