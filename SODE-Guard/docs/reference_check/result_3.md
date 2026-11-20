# Reference check: refs_3.bib (16 entries)

Method note: only web search worked in this session. WebFetch and direct curl could not reach crossref.org, datacite.org, doi.org, dblp.org or kaggle.com because the session's network does not allow those hosts. So DOI checks rest on search-indexed publisher, DBLP, PMLR, CVF, USENIX, SciTePress and repository records, not on resolving each DOI directly.

| key | status | problem found | corrected BibTeX fields (exact) | source URL |
|---|---|---|---|---|
| anaedevha2025ics3d | UNVERIFIABLE | The dataset exists on Kaggle under user `rogernickanaedevha`, and the title "Integrated Cloud Security 3Datasets (ICS3D)" matches. Two things could not be checked. (a) The DOI 10.34740/KAGGLE/DSV/12483891: kaggle.com and datacite were unreachable, and search returned no hit for the DOI. (b) The 11-author list: the Kaggle citation panel could not be read. A citing paper (EAI Endorsed Trans. IoT, 2026) credits the dataset to "Anaedevha RN" alone. The co-authors in the .bib (Freitas, Kalajdjieski, Gharib, McCann / Aly Amin, Dogan, Kuru, Sever, Angin / Ferrag) look like the authors of the source datasets (GUIDE, the container dataset, Edge-IIoT), which the Kaggle page says it combines. Check them against Kaggle's own "Cite" box before submission. Year 2025 not confirmed. Author's own work. | Suggested addition (verified): `url = {https://www.kaggle.com/datasets/rogernickanaedevha/integrated-cloud-security-3datasets-ics3d}`. Keep the author list and DOI only if they match Kaggle's Cite box exactly. | https://www.kaggle.com/datasets/rogernickanaedevha/integrated-cloud-security-3datasets-ics3d ; https://publications.eai.eu/index.php/IoT/article/view/15122 |
| anaedevha2026fokker | UNVERIFIABLE | Author's own; partly findable. Roger Nick Anaedevha's ResearchGate profile lists a conference paper dated June 2026 with Alexander G. Trofimov, titled "Uncertainty-Aware Network Intrusion Detection Using Fokker-Planck Moment Propagation and Stochastic Dynamics in Temporal Graph Neural Networks". That matches the title and authors. The conference "IEEE 27th International Conference of Young Professionals in Electron Devices and Materials (EDM)" exists (the 26th was in 2025). The DOI 10.1109/EDM69524.2026.11632112 and pages 1--11 could not be checked: IEEE Xplore and Crossref were not indexed or not reachable. Note that EDM papers are usually short, so check "1--11" against Xplore. | Optional, to match IEEE style: `booktitle = {2026 IEEE 27th International Conference of Young Professionals in Electron Devices and Materials (EDM)}`. The DOI and pages cannot be verified; check them on IEEE Xplore. | https://www.researchgate.net/profile/Roger-Nick-Anaedevha ; https://www.conferencelists.org/?p=273233 |
| anaedevha2026mstgnntia | UNVERIFIABLE | Not found online (author's own; cannot verify). The exact-title search found nothing on IEEE Xplore or elsewhere. This is consistent with the "Accepted, to appear" note. | none | (none found) |
| andriushchenko2020square | OK | Title, the 4 authors in order, ECCV 2020 and pp. 484--501 all match. The published DOI is 10.1007/978-3-030-58592-1_29 (optional to add). | Optional: `doi = {10.1007/978-3-030-58592-1_29}` | https://infoscience.epfl.ch/record/273123 ; https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680477.pdf |
| araujo2023sll | OK | Title, the 5 authors in order, and ICLR 2023 (oral) all match. | none | https://iclr.cc/virtual/2023/poster/11520 ; https://arxiv.org/abs/2303.03169 |
| breiman2001rf | OK | Machine Learning, 45(1):5--32, 2001 matches. The published title is "Random Forests" (the case is handled by the bib style). | Optional: `number = {1}`, `doi = {10.1023/A:1010933404324}` | https://dx.doi.org/10.1023/A%3A1010933404324 ; https://mlanthology.org/mlj/2001/breiman2001mlj-random |
| cohen2019smoothing | OK | ICML 2019, PMLR 97, pp. 1310--1320, and the authors all match. | none | https://proceedings.mlr.press/v97/cohen19c.html |
| engelen2021troubleshooting | OK | Title, the 3 authors and the 2021 IEEE SPW all match. Pages and DOI are available but missing. | Optional: `pages = {7--12}`, `doi = {10.1109/SPW53761.2021.00009}` | https://dblp.uni-trier.de/rec/conf/sp/EngelenRJ21.html ; https://researchr.org/publication/EngelenRJ21 |
| halbouni2022cnnlstm | OK | Title, the 6 authors in order, and IEEE Access 10:99837--99849 (2022) match per DBLP. DOI 10.1109/ACCESS.2022.3206425 is given for this paper by Semantic Scholar-indexed citing references (not resolved directly). | none | https://dblp.org/rec/journals/access/HalbouniGHHKA22 ; https://irep.iium.edu.my/100333/ ; https://www.semanticscholar.org/paper/f88008cc769fb7c6ad48f3a6905248b5f3aa6048 |
| kang2021sodef | OK | Title, the 4 authors in order, and NeurIPS 2021 (Advances in NeurIPS 34) all match. | none | https://proceedings.neurips.cc/paper/2021/hash/7d5430cf85f78c4b7aa09813b14bce0d-Abstract.html |
| liu2020sdenoise | OK | Title, the 6 authors in order, and CVPR 2020 all match. No pages are given in the .bib. CVF lists pp. 282--290; researchr lists 279--287. | Optional: `pages = {282--290}` (per CVF Open Access), `doi = {10.1109/CVPR42600.2020.00036}` (per mlanthology; not resolved directly) | https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_How_Does_Noise_Help_Robustness_Explanation_and_Exploration_under_the_CVPR_2020_paper.html |
| menon2021logit | OK | Title, the 6 authors in order, and ICLR 2021 all match. | none | https://iclr.cc/virtual/2021/poster/2675 ; https://arxiv.org/abs/2007.07314 |
| pendlebury2019tesseract | OK | Title, the 5 authors in order, and USENIX Security 2019 all match. Pages are available. | Optional: `pages = {729--746}` | https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury |
| sharafaldin2018cicids | OK | Title, the 3 authors and ICISSP 2018 all match. Pages and DOI are available. | Optional: `pages = {108--116}`, `doi = {10.5220/0006639801080116}` | https://www.scitepress.org/Papers/2018/66398/ |
| vitorino2024advbench | FIX | The author list is truncated to "and others". The full list is João Vitorino, Miguel Silva, Eva Maia, Isabel Praça. arXiv:2402.16912 is correct, but the paper was published at FPS 2023 (Springer, Foundations and Practice of Security), DOI 10.1007/978-3-031-57537-2_1, pp. 3--17 per ACM DL. Springer metadata gives 2024. Cite the published version, or at least list all authors. | `author = {Vitorino, Jo{\~a}o and Silva, Miguel and Maia, Eva and Pra{\c{c}}a, Isabel}` (plus the venue fields in the corrected entry below) | https://arxiv.org/abs/2402.16912 ; https://link.springer.com/chapter/10.1007/978-3-031-57537-2_1 ; https://dl.acm.org/doi/10.1007/978-3-031-57537-2_1 |
| zhang2019trades | OK | ICML 2019, PMLR 97, pp. 7472--7482, and the 6 authors in order all match. The full names are "Xing, Eric P." and "Jordan, Michael I." (optional). | Optional: `Xing, Eric P.`, `Jordan, Michael I.` | https://proceedings.mlr.press/v97/zhang19p.html |

## Corrected full BibTeX for FIX rows

Option A (preferred): the published version. The LNCS volume number was not verified, so it is left out.

```bibtex
@inproceedings{vitorino2024advbench,
  author    = {Vitorino, Jo{\~a}o and Silva, Miguel and Maia, Eva and Pra{\c{c}}a, Isabel},
  title     = {An adversarial robustness benchmark for enterprise network intrusion detection},
  booktitle = {Foundations and Practice of Security (FPS 2023)},
  publisher = {Springer},
  pages     = {3--17},
  year      = {2024},
  doi       = {10.1007/978-3-031-57537-2_1}
}
```

Option B: keep the arXiv citation, with the full author list.

```bibtex
@article{vitorino2024advbench,
  author  = {Vitorino, Jo{\~a}o and Silva, Miguel and Maia, Eva and Pra{\c{c}}a, Isabel},
  title   = {An adversarial robustness benchmark for enterprise network intrusion detection},
  journal = {arXiv preprint arXiv:2402.16912},
  year    = {2024}
}
```

## Items to check by hand before submission (could not be verified here)
- anaedevha2025ics3d: copy the author list and DOI exactly from the Kaggle dataset page's "Cite" box.
- anaedevha2026fokker: confirm the DOI, the pages (1--11) and the exact proceedings title on IEEE Xplore.
- anaedevha2026mstgnntia: no online record; leave it as "accepted, to appear" and add the DOI once it is published.
