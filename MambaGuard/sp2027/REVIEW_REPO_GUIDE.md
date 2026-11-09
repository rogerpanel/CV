# Publishing the anonymised artifact for reviewers

The package is `sp2027/dist/review-artifact.zip` (built by
`bash sp2027/build_artifact.sh`). It already passed the identity scan, the
stale-claim scan and the test suite inside the copy. Every folder holds at
most 100 files, so it can be uploaded through the GitHub website.

A GitHub repository is **not** anonymous by itself: its URL shows the account
name. Reviewers should get the link from Anonymous GitHub
(anonymous.4open.science), which mirrors the repository with identifying
words removed.

## Step 1: create the repository

1. github.com → **New repository**.
2. Choose a neutral name with no personal, institutional or project words (not "MambaGuard": that name
   links to your public model repository), e.g. `agent-protocol-detectors`.
3. Leave the README, .gitignore and licence boxes **unticked**; the package
   has its own. Private or Public both work.

## Step 2: put the files in it (choose one)

**A. Let Claude push it.** Tell Claude the repository name in the chat.
It can attach the repository to the session and push the package as one
commit with an anonymous author.

**B. Browser upload.**
1. Unzip `review-artifact.zip` on your computer.
2. On the empty repository's page, click **uploading an existing file**.
3. Open the unzipped `review-artifact` folder and select **its
   contents**, not the folder itself.
4. Drag them in. If GitHub refuses more than 100 files, upload one
   top-level folder at a time: first the root files (`README.md`,
   `LICENSE`, `setup.py`, ...), then `apdet/`, then `reproducibility/`,
   and so on.
5. Commit each batch to `main`.

**C. Command line.**
```bash
unzip review-artifact.zip && cd review-artifact
git init -b main && git add -A
git -c user.name=Anonymous -c user.email=anonymous@example.com commit -m "Artifact"
git remote add origin https://github.com/<account>/<repo>.git && git push -u origin main
```

## Step 3: make the anonymous link

1. Go to anonymous.4open.science and sign in with GitHub.
2. Choose **Anonymize your repository** and pick the repository and branch
   `main`.
3. Under terms to remove, add your names, your GitHub user name, your
   institution, and `MambaGuard`.
4. Set the repository ID to `agent-protocol-detectors` (the paper's Appendix I already uses
   `https://anonymous.4open.science/r/agent-protocol-detectors`; if that ID is taken, pick another and
   tell Claude to update the paper), set the expiry after the review period, and save.
5. Open the link in a private browser window and check the README renders
   and no name appears.
6. Put the link in the paper (Appendix I, `\url{...}`) and in the
   submission form's artifact field.
7. Optionally, put the mirror's **Download** link into `ARTIFACT_ZIP_URL` in
   `kaggle/run_on_kaggle.ipynb`, so reviewers can run the study on Kaggle
   without uploading anything.

## Getting the Kaggle results zip (for your own records)

The editor view of a notebook loses `/kaggle/working` whenever its session
restarts. Only a **committed version** keeps its files.

1. On the notebook's page, open the version list (the "Version N" menu
   next to the notebook title, or the *Versions* tab).
2. Look for a version whose run took about 3 hours and finished
   successfully.
3. Open it and scroll to **Output**. `results_v3_all.zip` is there; click
   the download icon next to it.

If no such version exists, the run happened only in the editor and its
files are gone. The paper does not need them: every number is in
`reproducibility/kaggle_gpu_v3/RESULTS_TRANSCRIBED.md`. To regenerate the
per-episode scores, run the notebook again through **Save Version → Save &
Run All (Commit)** (about 3 hours).
