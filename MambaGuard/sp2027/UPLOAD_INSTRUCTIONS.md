# Publishing the anonymised artifact

anonymous.4open.science does not take a zip upload. It mirrors a GitHub
repository and hides the owner, so the zip has to go into a GitHub repo first.

## 1. Put the zip contents in a fresh GitHub repo

Use a new repository with a neutral name (for example `mg-artifact`). It can
be private. Do not reuse a repository whose history or name identifies you.

```bash
unzip mambaguard-artifact.zip
cd mambaguard-artifact
git init -b main
git add .
git -c user.name="Anonymous" -c user.email="anonymous@example.com" \
    commit -m "Artifact for anonymous review"
git remote add origin https://github.com/<your-account>/mg-artifact.git
git push -u origin main
```

The neutral author name keeps your identity out of the commit metadata.

## 2. Create the anonymous mirror

1. Go to https://anonymous.4open.science and sign in with GitHub.
2. Choose to anonymise a repository and select `mg-artifact`, branch `main`.
3. In the list of terms to hide, add your GitHub username, your surnames, your
   institution and your e-mail domain as a second line of defence. The build
   script has already removed them, so this should hide nothing.
4. Set the expiry to after the notification date (5 March 2027).
5. Submit. You get a link of the form `https://anonymous.4open.science/r/<ID>`.

## 3. Put the link in the paper

Replace the red `\url{https://anonymous.4open.science/r/<ID>}` in the
"Artifact Availability" appendix of `MambaGuard_sp2027.tex` with your link.
Open the link in a private browser window and check that no name, account or
affiliation appears anywhere.

## 4. Rebuild and freeze before 17 November

The mirror follows the GitHub repo, so the link stays the same when you push
updates. After the GPU runs finish:

```bash
cd MambaGuard
bash sp2027/build_artifact.sh        # rebuilds sp2027/dist/mambaguard-artifact.zip
```

Copy the new contents into `mg-artifact`, commit with the same neutral author,
and push. Do this before the paper deadline, then stop pushing: the S&P 2027
call for papers does not allow artifact updates after the deadline.

## What `build_artifact.sh` checks

- Runs `anonymize.sh`, then removes compiled caches, the legacy IEEE Access
  scripts and results, the docs and notebooks, and the dataset manifest that
  pointed at an author-owned Zenodo record.
- Rewrites the platform-specific taxonomy and dataset names and removes
  author-owned DOIs.
- Fails if any author name, affiliation, account, DOI, e-mail fragment or the
  related-submission names appear anywhere in the copy.
- Fails if the old headline numbers or "previous version" wording reappear.
- Runs the full test suite inside the anonymised copy.
