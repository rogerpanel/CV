# Reference check: refs_4.bib (15 entries)

How this was checked: only web search was available. Direct fetching was blocked in this session (doi.org, api.crossref.org, link.springer.com, dblp.org, kaggle.com, eng.mephi.ru all refused), so no DOI was resolved directly. DOIs were matched through search-indexed publisher, dblp, arXiv and ResearchGate records that show the DOI next to the title.

| key | status | problem found | corrected BibTeX fields (exact) | source URL |
|---|---|---|---|---|
| anaedevha2025iis3d | OK (caveat) | This is the author's own dataset. Third-party papers cite it as "Integrated IDPS Security 3Datasets (IIS3D)", Kaggle, 2025, and a cached Kaggle listing appeared in search. I could not open the Kaggle page, so I did not confirm the URL slug or whether there are co-authors (one citing paper writes "Anaedevha, R.N., et al."). A citing source gives the DOI 10.34740/KAGGLE/DSV/12479689. Confirm it on the Kaggle page before adding it. One Springer paper describes a similar dataset as "Integrated Cloud Security 3Datasets (ICS3D)", which looks like a separate dataset. Make sure the right one is cited. | Optional, only after checking on Kaggle: `doi = {10.34740/KAGGLE/DSV/12479689}` | https://link.springer.com/chapter/10.1007/978-3-032-07690-8_35 (citing source); https://www.kaggle.com/datasets/rogernickanaedevha/integrated-idps-security-3datasets (not reachable from here) |
| anaedevha2026hgp | OK | Title, authors (Roger Nick Anaedevha, Alexander G. Trofimov, Yuri V. Borodachev), journal Neurocomputing, vol. 677, article 133105, year 2026 and DOI all match the dblp/ACM records. | none | https://dblp.org/db/journals/ijon/ijon677.html |
| anaedevha2026ode | OK | Title, authors, journal and DOI match the Springer article page as shown in search results. It was published 11 Apr 2026. The entry has no volume or article number. The search results give vol. 12, article 165. | Optional: `volume = {12}`, `pages = {165}` | https://link.springer.com/article/10.1007/s40747-026-02291-7 |
| anil2019sorting | OK | Matches PMLR 97:291–301 (ICML 2019), authors Cem Anil, James Lucas, Roger Grosse. The official title is "Sorting Out Lipschitz Function Approximation". The sentence-case version is only a style difference. | none | https://proceedings.mlr.press/v97/anil19a.html |
| arp2022dos | OK | Title, all 8 authors in the right order, and USENIX Security 2022 (31st) all match. The pages were not given. They are 3971–3988. | Optional: `pages = {3971--3988}` | https://www.usenix.org/conference/usenixsecurity22/presentation/arp |
| carlini2019evaluating | OK | arXiv:1902.06705 (2019). The PDF lists 9 authors, ending with Kurakin, in the same order as the entry. Some index pages show only the first 8. | none | https://arxiv.org/abs/1902.06705 |
| croce2020autoattack | OK | Matches PMLR 119:2206–2216 (ICML 2020), authors Croce and Hein. | none | https://proceedings.mlr.press/v119/croce20b.html |
| ennaji2024advsurvey | FIX (minor) | The paper exists (arXiv:2409.18736, 2024) and the title is correct. The author list is cut short to "Ennaji, Sabrine and others". The full list is Sabrine Ennaji, Fabio De Gaspari, Dorjan Hitaj, Alicia Kbidi, Luigi V. Mancini. | `author = {Ennaji, Sabrine and De Gaspari, Fabio and Hitaj, Dorjan and Kbidi, Alicia and Mancini, Luigi V.}` | https://arxiv.org/abs/2409.18736 ; https://papers.cool/arxiv/2409.18736 |
| holm1979 | OK | Matches Scand. J. Statist. 6(2):65–70, 1979, author Sture Holm. The JSTOR DOI 10.2307/4615733 can be added if you want one. | Optional: `doi = {10.2307/4615733}` | https://www.jstor.org/stable/4615733 (via bibbase/BCO-DMO records) |
| ke2017lightgbm | OK | All 8 authors and their order match the NeurIPS 2017 proceedings PDF ("Finley" is the correct spelling). The volume and pages were not given. They are vol. 30, pp. 3146–3154. The pages come from the mlanthology BibTeX only. | Optional: `volume = {30}`, `pages = {3146--3154}` | https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf |
| lo2022egraphsage | OK | Title, the 5 authors in order, and the venue (IEEE/IFIP NOMS 2022, Budapest) match. The arXiv record (2103.16329) links the DOI 10.1109/NOMS54207.2022.9789878. I could not open the IEEE Xplore record to confirm pages 1–9. | none (pp. 1–9 not checked against Xplore) | https://arxiv.org/abs/2103.16329 |
| mosaiyebzadeh2025graphmamba | FIX (major) | The paper exists, but the authors, volume, article number, DOI and year are all wrong. The real authors are Safa Ben Atitallah, Maha Driss, Wadii Boulila (ScienceDirect, dblp, ResearchGate). Mosaiyebzadeh is not an author and wrote a different paper (TNN-IDS). It appears in Computer Networks vol. 275 (Feb 2026), article 111933, DOI 10.1016/j.comnet.2025.111933. The cited DOI ...2025.108989 has no record. It looks derived from the ScienceDirect PII S1389128625008989. The publisher title writes "IDS–GraphMamba" with an en dash. | `author = {Ben Atitallah, Safa and Driss, Maha and Boulila, Wadii}`, `volume = {275}`, `pages = {111933}`, `year = {2026}`, `doi = {10.1016/j.comnet.2025.111933}`, `title = {{IDS--GraphMamba}: A {Markov}-enhanced graph {Mamba} framework for real-time intrusion detection in {IoMT} edge networks}`; **the citation key should be renamed too** (e.g. `benatitallah2026graphmamba`) and the in-text mentions of "Mosaiyebzadeh et al." changed | https://www.sciencedirect.com/science/article/abs/pii/S1389128625008989 ; https://www.researchgate.net/publication/398578485 |
| rice2020overfitting | OK | Matches ICML 2020, authors Rice, Wong, Kolter. The volume and pages were not given. They are PMLR 119:8093–8104. | Optional: `series = {PMLR}`, `volume = {119}`, `pages = {8093--8104}` | https://proceedings.mlr.press/v119/rice20a.html |
| simonetto2024tabularbench | OK | Title, authors (Simonetto, Ghamizi, Cordy) and venue (NeurIPS 2024 Datasets and Benchmarks Track) match. The volume was not given. It is 37. | Optional: `volume = {37}` | https://proceedings.neurips.cc/paper_files/paper/2024/hash/8f5defaa8dc728c3e4aaee59ac8ca942-Abstract-Datasets_and_Benchmarks_Track.html |
| wang2023bars | OK | Title, the 7 authors in order and NDSS 2023 all match. | none | https://www.ndss-symposium.org/ndss-paper/bars-local-robustness-certification-for-deep-learning-based-traffic-analysis-systems/ |

## Corrected full BibTeX entries for FIX rows

```bibtex
@article{ennaji2024advsurvey,
  author  = {Ennaji, Sabrine and De Gaspari, Fabio and Hitaj, Dorjan and Kbidi, Alicia and Mancini, Luigi V.},
  title   = {Adversarial challenges in network intrusion detection systems: Research insights and future prospects},
  journal = {arXiv preprint arXiv:2409.18736},
  year    = {2024}
}

@article{benatitallah2026graphmamba,
  author  = {Ben Atitallah, Safa and Driss, Maha and Boulila, Wadii},
  title   = {{IDS--GraphMamba}: A {Markov}-enhanced graph {Mamba} framework for real-time intrusion detection in {IoMT} edge networks},
  journal = {Computer Networks},
  volume  = {275},
  pages   = {111933},
  year    = {2026},
  doi     = {10.1016/j.comnet.2025.111933}
}
```

For the GraphMamba entry, if you keep the old key `mosaiyebzadeh2025graphmamba` so that `\cite` commands still work, still check the paper text for any "Mosaiyebzadeh et al." wording.

## Notes

- The volume, article number and DOI for GraphMamba come from several search-indexed records (ScienceDirect, ResearchGate, citing papers). I could not resolve the DOI on doi.org myself because the host was blocked.
- The fields marked "Optional" are not errors. They fill in information the entry leaves out.
