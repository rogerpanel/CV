# Upload guide: GitHub repository, anonymous mirror for review, and submission files

## 0. What is in this folder

| File | Use |
|---|---|
| `bc-grade-opendata_v1.0.2_github.zip` | The full experimentation package (code, configs, tests, results, manuscript sources). **Identity-free**: safe for the anonymous mirror. This is what goes on GitHub. |
| `manuscript_v1.0.2_overleaf.zip` | Overleaf project: named + blinded manuscript, title page, highlights, graphical abstract (contains your details). |
| `submission/` | Ready-to-upload PDFs for *Computers in Biology and Medicine* (see §4). |
| `manuscript_v1.0.2.tex`, `manuscript_v1.0.2_anonymised.tex` | Single-file (flattened) LaTeX of each version; compile next to the `generated/data` folder from the Overleaf zip. |
| `ASSESSMENT.md`, `anchors_verification.md` | Internal notes (not for the repository). |

## 1. Create the GitHub repository (5 minutes)

1. On GitHub: **New repository** → name e.g. `bc-grade-opendata` → *Public* → do **not** add a README,
   licence or .gitignore (the package already has them) → **Create**.
   (A username that reveals your identity is fine: the anonymous mirror in step 3 hides it.)
2. Upload with Git (the browser uploader is limited to 100 files at a time; the package has ~130):
   ```bash
   unzip bc-grade-opendata_v1.0.2_github.zip
   cd bc-grade-opendata
   git init -b main
   git add .
   git commit -m "bc-grade-opendata v1.0.2 (submission build)"
   git remote add origin https://github.com/<your-username>/bc-grade-opendata.git
   git push -u origin main
   ```
   No Git installed? GitHub Desktop → *File › Add local repository* → select the unzipped folder →
   *Publish repository*.
3. Check **Actions**: the `data-and-guards` workflow downloads the open data, verifies checksums and
   runs the 12 tests (≈ 3 min). A green tick is what reviewers will see.

## 2. Reproduce locally (optional but recommended before submission)
```bash
pip install -r requirements.txt
make all            # ≈ 1 h on 4 cores; regenerates every number; `git diff paper/generated/numbers.tex` should be empty
```

## 3. Anonymous mirror for double-anonymised review

1. Go to <https://anonymous.4open.science> → *Login with GitHub* → **Anonymize your repository**.
2. Repository: `https://github.com/<your-username>/bc-grade-opendata`, branch `main`.
3. *Terms to anonymize* (one per line): your GitHub username, `Ajibola`, `Aminat`, `Hafr`, `uhb`.
   (The package contains none of these; the list is a safety net.)
4. Options: enable *download* (reviewers can fetch the whole repository as a ZIP), set an *expiration*
   after the expected review period, keep *update* on so later pushes propagate.
5. Copy the link (`https://anonymous.4open.science/r/<id>/`). Put it in `paper/main.tex`:
   `\newcommand{\CodeURL}{\url{https://anonymous.4open.science/r/<id>}}` → recompile (Overleaf or
   `make paper`) → re-export the PDFs in §4. Push the same edit to GitHub.

## 4. Submitting to *Computers in Biology and Medicine* (Elsevier; Q1; hybrid, so the subscription route has no APC)

Upload, in Editorial Manager:
- **Cover letter:** `submission/cover_letter.pdf` (editable copy: `submission/cover_letter.docx`; fill the date, repository link and optional suggested reviewers).
- **Manuscript (anonymised):** `submission/manuscript_anonymised.pdf` (line-numbered, no names; supplement included at the end).
- **Title page (separate file):** `submission/title_page.pdf`.
- **Highlights:** `submission/highlights.pdf` (5 bullets, ≤ 85 characters each).
- **Graphical abstract:** `submission/graphical_abstract.pdf` (Figure 1).
- **LaTeX source** (requested at revision/acceptance): `manuscript_v1.0.2_overleaf.zip`.
- Choose the subscription (non open-access) publishing option to avoid the APC.
- Before submitting, check the current *Guide for Authors* (abstract length, review model, reference style)
  and fill the `[Confirm]` placeholders on the title page (funding, competing interests, AI-use statement).

Backups if needed (also Elsevier hybrid, same template): *Computer Methods and Programs in Biomedicine*;
*International Journal of Medical Informatics*.

## 5. After acceptance (not before: a DOI and named repository would break anonymity)
1. Fill `CITATION.cff` (authors, ORCID), set `\CodeURL` to the GitHub URL, rebuild.
2. Zenodo: log in to <https://zenodo.org> **with GitHub** → *GitHub* tab → switch the repository **on** →
   create a GitHub *Release* `v1.0.0` → Zenodo mints the DOI automatically. No API key is needed, and
   API tokens should never be pasted into chats, issues or code.
3. Add the DOI to `paper/main.tex` (Code availability) and `CITATION.cff`.

## 6. Optional TabPFN run (needs Hugging Face access)
Accept the licence at <https://huggingface.co/Prior-Labs>, create a read token, then on your machine:
`export HF_TOKEN=hf_...` and `make tabpfn`. The tables, figures and the Limitations sentence update
automatically; push the regenerated files.
