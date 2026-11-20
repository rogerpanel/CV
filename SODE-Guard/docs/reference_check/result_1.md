# Reference check: refs_1.bib

Method note: In this session only WebSearch worked. WebFetch, curl, Crossref, doi.org and dblp.org were all blocked by the session network ("host not allowed"). So every check below rests on search-engine snippets and indexed publisher, DBLP, PMLR, CVF and NeurIPS pages, not on fetched pages. Where a field showed up only in one snippet, the row says so.

| key | status | problem found | corrected BibTeX fields (exact) | source URL |
|---|---|---|---|---|
| amin2024misuse | FIX | The paper exists: AINA 2024, pp. 217–226, Springer Nature Switzerland, 2024. The author names are split wrongly for BibTeX. As written, "Aly Amin, Mohamed" makes "Aly Amin" the surname, and "Harun Dogan" and "Sena Kuru" become surnames too. It would print as "M. Aly Amin, A. Harun Dogan, E. Sena Kuru". DBLP's key AminDKSA24 shows the surnames are Amin, Dogan, Kuru, Sever and Angin. The DOI is missing. The booktitle has no edition or year. | `author = {Amin, Mohamed Aly and Dogan, Adnan Harun and Kuru, Elif Sena and Sever, Yigit and Angin, Pelin}`, `booktitle = {Advanced Information Networking and Applications (AINA 2024)}`, `doi = {10.1007/978-3-031-57942-4_22}` (library catalogues put the book in the series Lecture Notes on Data Engineering and Communications Technologies, vol. 204, but I could not confirm that number on Springer, so it is not added) | https://link.springer.com/chapter/10.1007/978-3-031-57942-4_22 ; https://dblp.dagstuhl.de/rec/conf/aina/AminDKSA24.html |
| anaedevha2025stochtx | OK | The DOI resolves to "Stochastic Multimodal Transformer with Uncertainty Quantification for Robust Network Intrusion Detection" by Roger Nick Anaedevha and Alexander Gennadievich Trofimov. It is in NEUROINFORMATICS 2025, Studies in Computational Intelligence vol. 1241, Springer, Cham, 2025. The title, authors, series, volume, year and DOI all match. Pages are not given in the entry (optional to add). | optional: `pages = {428--447}` | https://link.springer.com/chapter/10.1007/978-3-032-07690-8_35 ; https://link.springer.com/book/10.1007/978-3-032-07690-8 |
| anaedevha2026mambashield | FIX | The paper exists: Expert Systems with Applications, vol. 308, article 131175, 2026, DOI 10.1016/j.eswa.2026.131175. The entry is missing the volume. Indexed listings spell the title "...sequential modelling" (British spelling), not "modeling". The Springer KAIS reference list cites it with 3 authors (Anaedevha, Trofimov, Borodachev), which matches the entry. One academia.edu listing shows only two authors, but that list may be cut off. I could not reach the ScienceDirect page itself, so check the author list and title spelling there (author's own paper). | `title = {{MambaShield}: Selective state-space models for poisoning-resilient sequential modelling}`, `volume = {308}` | https://independent.academia.edu/RAnaedevha ; https://link.springer.com/article/10.1007/s10115-026-02824-z (reference list) |
| anaedevha2026pacbayes | OK | The title matches exactly. Springer and ACM DL list the authors as Roger Nick Anaedevha, Alexander Gennadievich Trofimov and Yuri Vladimirovich Borodachev. It is Knowl Inf Syst 68, 232 (2026), published 3 Aug 2026, DOI 10.1007/s10115-026-02824-z. Caveat: one search summary gave a two-author "cite this article" line. Two other searches, including the ACM DL record, gave all three authors. The author should check the Springer page once. | none | https://link.springer.com/article/10.1007/s10115-026-02824-z ; https://dl.acm.org/doi/10.1007/s10115-026-02824-z |
| anon_ics3d | UNVERIFIABLE | Anonymised self-citation for a dataset, withheld for review. No public dataset named "ICS3D" was found. That is expected, and fine for double-blind review as long as the venue allows anonymised self-citations. | none | n/a |
| athalye2018obfuscated | OK | ICML 2018, PMLR 80:274–283. Authors Athalye, Carlini, Wagner. Title matches. | none | https://proceedings.mlr.press/v80/athalye18a.html |
| chen2016xgboost | OK | KDD 2016 (22nd ACM SIGKDD). Authors Chen, Guestrin. Title matches. Pages and DOI are not given (optional to add). | optional: `pages = {785--794}`, `doi = {10.1145/2939672.2939785}` | https://arxiv.org/abs/1603.02754 ; https://researchr.org/publication/ChenG16-15/bibtex |
| cui2019classbalanced | OK | CVPR 2019. Authors Cui, Jia, Lin, Song, Belongie, as on the CVF Open Access page. Title matches. | optional: `pages = {9268--9277}` | https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html |
| grinsztajn2022trees | OK | NeurIPS 2022 Datasets and Benchmarks Track. Authors Grinsztajn, Oyallon, Varoquaux. The title with "typical" is the NeurIPS title. The arXiv title leaves out "typical". | none | https://nips.cc/virtual/2022/poster/55627 ; https://papers.nips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf |
| huang2025mars | FIX | (a) The authors are surnames only ("Huang and Lin and Chen"). Full names: Mengdie Huang, Yingjun Lin, Xiaofeng Chen, Elisa Bertino. (b) Wrong title. The ACM TOPS article at DOI 10.1145/3715121 is titled "Dimensional Robustness Certification for Deep Neural Networks in Network Intrusion Detection Systems". "MARS" is the method name. The earlier TrustCom 2024 conference version is titled "MARS: Robustness Certification for Deep Network Intrusion Detectors via Multi-Order Adaptive Randomized Smoothing". The title in the entry matches neither. (c) The volume is missing: vol. 28, no. 3 (Aug 2025). The article number 39 and pages 39:1–39:39 come from a DBLP TOPS vol. 28 snippet only, so check them on the ACM page. | `author = {Huang, Mengdie and Lin, Yingjun and Chen, Xiaofeng and Bertino, Elisa}`, `title = {Dimensional Robustness Certification for Deep Neural Networks in Network Intrusion Detection Systems}`, `volume = {28}`, `number = {3}`, `articleno = {39}`, `pages = {39:1--39:39}` | https://dl.acm.org/doi/10.1145/3715121 ; https://dblp.uni-trier.de/db/journals/tissec/tissec28.html ; https://www.semanticscholar.org/paper/d120dd3d7670ac16c43496349f8e615d2ed5dcea |
| li2020scalable | OK | AISTATS 2020, PMLR 108:3870–3882. Authors Li, Wong, Chen, Duvenaud. Title matches. Not to be confused with the PMLR 118 paper "Scalable Gradients and Variational Inference for SDEs". | none | https://proceedings.mlr.press/v108/li20i.html |
| madry2018pgd | OK | ICLR 2018. Five authors in the correct order. Title matches. | optional: `url = {https://openreview.net/forum?id=rJzIBfZAb}` | https://iclr.cc/virtual/2018/poster/67 ; https://arxiv.org/abs/1706.06083 |
| moustafa2015unsw | FIX | The paper exists: MilCIS 2015, pp. 1–6, Moustafa and Slay. The IEEE title is incomplete in the entry. It is "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)". Pages and DOI are missing. | `title = {{UNSW-NB15}: A comprehensive data set for network intrusion detection systems ({UNSW-NB15} network data set)}`, `pages = {1--6}`, `doi = {10.1109/MilCIS.2015.7348942}` | https://www.semanticscholar.org/paper/0e7af8e91b8cb2cea1164be5ac5d280b0d12c153 ; https://dblp1.uni-trier.de/rec/conf/milcis/MoustafaS15.html |
| sadinle2019lac | OK | JASA 114(525):223–234, 2019. Authors Sadinle, Lei, Wasserman. Title matches. | optional: `doi = {10.1080/01621459.2017.1395341}` | https://ideas.repec.org/a/taf/jnlasa/v114y2019i525p223-234.html |
| tsuzuku2018lipschitz | OK | NeurIPS 2018 (Advances in Neural Information Processing Systems 31). Authors Tsuzuku, Sato, Sugiyama. Title matches. | optional: `volume = {31}`, `pages = {6541--6550}` | https://papers.nips.cc/paper/2018/hash/485843481a7edacbfce101ecb1e4d2a8-Abstract.html |
| wu2022rtids | OK | IEEE Access 10:64375–64387, 2022. Authors Wu, Zhang, Wang, Sun. The DOI 10.1109/ACCESS.2022.3182333 matches. | none | https://doaj.org/article/13b8fab7f30846998bf80366d8d74ca9 |

## Corrected full BibTeX entries (FIX rows)

```bibtex
@inproceedings{amin2024misuse,
  author    = {Amin, Mohamed Aly and Dogan, Adnan Harun and Kuru, Elif Sena and Sever, Yigit and Angin, Pelin},
  title     = {Misuse detection and response for orchestrated microservices based software},
  booktitle = {Advanced Information Networking and Applications (AINA 2024)},
  pages     = {217--226},
  publisher = {Springer Nature Switzerland},
  year      = {2024},
  doi       = {10.1007/978-3-031-57942-4_22}
}

@article{anaedevha2026mambashield,
  author  = {Anaedevha, Roger Nick and Trofimov, Alexander G. and Borodachev, Yuri V.},
  title   = {{MambaShield}: Selective state-space models for poisoning-resilient sequential modelling},
  journal = {Expert Systems with Applications},
  volume  = {308},
  pages   = {131175},
  year    = {2026},
  doi     = {10.1016/j.eswa.2026.131175}
}

@article{huang2025mars,
  author  = {Huang, Mengdie and Lin, Yingjun and Chen, Xiaofeng and Bertino, Elisa},
  title   = {Dimensional Robustness Certification for Deep Neural Networks in Network Intrusion Detection Systems},
  journal = {ACM Transactions on Privacy and Security},
  volume  = {28},
  number  = {3},
  articleno = {39},
  pages   = {39:1--39:39},
  year    = {2025},
  doi     = {10.1145/3715121}
}

@inproceedings{moustafa2015unsw,
  author    = {Moustafa, Nour and Slay, Jill},
  title     = {{UNSW-NB15}: A comprehensive data set for network intrusion detection systems ({UNSW-NB15} network data set)},
  booktitle = {Military Communications and Information Systems Conference (MilCIS)},
  pages     = {1--6},
  year      = {2015},
  doi       = {10.1109/MilCIS.2015.7348942}
}
```

Notes:
- huang2025mars: if the paper means to cite the method by name, keep "MARS" in the running text. The bibliography title must be the published one. Check `articleno`/`pages` (39) on the ACM page; I saw them only in a DBLP snippet. If the paper instead means the TrustCom 2024 conference version, cite "MARS: Robustness Certification for Deep Network Intrusion Detectors via Multi-Order Adaptive Randomized Smoothing", TrustCom 2024, pp. 767–774. That is a different reference.
- anaedevha2026mambashield: I could not reach the ScienceDirect page. Check the "modelling" spelling and the three-author list there.
- anaedevha2026pacbayes: three authors in the entry. Check once on Springer, because one snippet showed a two-author citation line.
