# Dual-Submission Audit --- MambaGuard / IEEE S&P 2027 Cycle 2

The S&P Call for Papers forbids concurrent submission of the same (or
substantially overlapping) work to any other peer-reviewed venue. This
worksheet is to be completed by the submitting author before 8 Nov 2026
and filed with the submission package. Every row must resolve to one of
`WITHDRAWN`, `NO OVERLAP`, `DIFFERENTIATED IN PAPER`, or `BLOCKER`.

Author: __________________________    Date filled: __________________

---

## Row 1 --- IEEE Access

- Manuscript title: ____________________________________________________
- Submission ID: ______________________________________________________
- Current status in the IEEE Access portal (as of today): _______________
- Withdrawal request sent?                 [ ] yes   [ ] no
- Date of withdrawal email: ___________________________________________
- Withdrawal acknowledgement received?     [ ] yes   [ ] no
- Acknowledgement filed at
  `sp2027/withdrawals/ieee_access.eml`?    [ ] yes   [ ] no
- Resolution:  [ ] WITHDRAWN     [ ] BLOCKER   (other states not valid)
- Note: a withdrawal request that has not been acknowledged in writing is
  not a withdrawal. If the acknowledgement does not arrive before 8 Nov,
  escalate to the IEEE Access EiC and treat the row as BLOCKER until it
  does.

---

## Row 2 --- IEEE TNNLS

- Manuscript title: ____________________________________________________
- The S&P-paper BibTeX entry currently labels this as a TNNLS submission.
- Current status in the TNNLS/ScholarOne portal: _______________________
- Overlap classification (pick one):
  - [ ] Full overlap --- same paper. Resolution must be WITHDRAWN.
  - [ ] Strict subset / superset --- a theorem, lemma, or table appears
        in both. Resolution must be WITHDRAWN unless the TNNLS version
        has already appeared, in which case see Row 3 rules.
  - [ ] No shared technical content, only shared background. Resolution
        may be NO OVERLAP; attach a one-paragraph justification below.
- Justification (if NO OVERLAP): ______________________________________
  ____________________________________________________________________
- If overlap, withdrawal email sent?        [ ] yes   [ ] no
- Date of withdrawal email: ___________________________________________
- Acknowledgement filed at
  `sp2027/withdrawals/tnnls.eml`?            [ ] yes   [ ] no
- Resolution:  [ ] WITHDRAWN   [ ] NO OVERLAP   [ ] BLOCKER

---

## Row 3 --- LipMamba-ICLR27

- The authors have a concurrent ICLR 2027 submission titled
  "LipMamba: Lipschitz theory for structured state-space models"
  (working title). It is **not** being withdrawn --- it is a separate
  paper with a different main contribution.
- Overlap area: Section 3's Lipschitz analysis of the SSM detector
  reuses two lemmas introduced in LipMamba.
- Overlap classification: **substantial but demarcated** (a theorem
  statement and a proof sketch are shared; the full proof is only in
  LipMamba).
- Disclosure plan on the S&P paper:
  - [ ] Cite LipMamba in the third person, as an anonymous preprint or
        a submitted-elsewhere reference; **do not** use first-person
        plural ("we show in LipMamba ...") that would deanonymise.
  - [ ] Add a footnote on the first page of Section 3 flagging that the
        two lemmas appear in a concurrently submitted anonymous
        manuscript and that the present paper's contribution is the
        application to MCP/A2A protocol graphs, not the Lipschitz theory
        itself.
  - [ ] Confirm the ICLR 2027 and S&P 2027 Cycle 2 anonymity windows do
        not conflict (ICLR 2027 openreview date: ______________).
  - [ ] Confirm with the ICLR co-authors (if non-overlapping co-author
        sets) that the S&P citation text is acceptable to them.
- Resolution:  [ ] DIFFERENTIATED IN PAPER   [ ] BLOCKER

---

## Row 4 --- Other concurrent submissions

List every other peer-reviewed venue (conference, journal, workshop with
archival proceedings) to which any version of this work, or a
substantial subset of it, has been submitted and is still under
consideration as of the S&P submission deadline. Preprint servers
(arXiv, OpenReview-as-preprint, SSRN) and non-archival workshops are
not conflicts and need not be listed here, but must still not
deanonymise.

| # | Venue | Title | Status | Overlap | Resolution        |
|---|-------|-------|--------|---------|-------------------|
| 1 |       |       |        |         | [ ] W [ ] N [ ] D |
| 2 |       |       |        |         | [ ] W [ ] N [ ] D |
| 3 |       |       |        |         | [ ] W [ ] N [ ] D |

Legend: W = WITHDRAWN, N = NO OVERLAP, D = DIFFERENTIATED IN PAPER.
Any row that resolves to BLOCKER halts submission until resolved.

---

## Sign-off

- [ ] All rows above resolve to WITHDRAWN / NO OVERLAP / DIFFERENTIATED.
- [ ] No row is BLOCKER as of the submission deadline.
- [ ] A PDF or printed copy of this file, filled in, is stored in
      `sp2027/dual_submission_audit_signed.pdf`.

Signature (submitting author): ________________________________________
