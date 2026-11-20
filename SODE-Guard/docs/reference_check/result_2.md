# Reference check: refs_2.bib (16 entries)

How this was checked: everything was verified with WebSearch results from publisher, proceedings, arXiv and indexing pages. This session could not fetch pages directly: WebFetch and curl to doi.org, api.crossref.org and link.springer.com were blocked by the session's network policy. As a result, **no DOI was resolved directly**. A DOI counts as matching only when a search result tied that exact DOI or URL to the paper.

| key | status | problem found | corrected BibTeX fields (exact) | source URL |
|---|---|---|---|---|
| anaedevha2024robust | OK | None. The Springer listing matches the title, the authors (R.N. Anaedevha, A.G. Trofimov), the journal, vol. 33, Suppl. 3, pp. S414–S423, and the 2024 volume year. The Springer article URL contains this exact DOI. Online-first date is 23 Jan 2025; the issue year 2024 is correct. | none | https://link.springer.com/article/10.3103/S1060992X24700681 |
| anaedevha2026estc | UNVERIFIABLE | Not found online (author's own; cannot verify). No record of this title turned up anywhere. The venue does exist: the IEEE 11th ESTC took place in Helsinki, 9–11 Sep 2026, with proceedings going to IEEE Xplore. If the paper has appeared in Xplore by now, add its DOI and pages. Otherwise keep "Accepted, to appear". | none (cannot verify) | https://www.conferencelists.org/event/ieee-11th-electronics-system-integration-technology-conference-estc/ ; https://eps.ieee.org/wp-content/uploads/2026/02/ESTC2026_CFP.pdf |
| anaedevha2026mstgnn | UNVERIFIABLE | Author's own; cannot be fully verified. One aggregator result lists "a multi-scale temporal graph neural network paper for cloud and industrial networks (IEEE EDM, 2026)" by Anaedevha. I found no IEEE Xplore record, the exact title string did not turn up, and the DOI 10.1109/EDM69524.2026.11632205 could not be resolved. The venue (IEEE 27th EDM) is listed. Before submitting, open the DOI by hand and check the pages (1--8). | none (cannot verify) | https://www.conferencelists.org/?p=273233 ; https://www.researchgate.net/profile/Roger-Nick-Anaedevha |
| anaedevha2026pqidps | OK | Verified: a TechRxiv preprint by Roger Nick Anaedevha, Alexander Gennadevich Trofimov and Yuri Vladimirovich Borodachev (MEPhI), posted January 2026 (listing dated 14 Jan 2026), describes PQIDPS as a hybrid classical-quantum ML framework for adversarially robust intrusion detection in post-quantum encrypted traffic. The DOI prefix 176836457 decodes to a 14 Jan 2026 timestamp, which fits. Not verified: the exact title string (an exact-phrase search returned nothing) and the DOI, which could not be opened. Click the DOI before submitting. | none | https://www.authorea.com/inst/26407?page=1&tag_filter=Signal+Processing+And+Analysis ; https://www.researchgate.net/profile/Roger-Nick-Anaedevha |
| anon_iis3d | UNVERIFIABLE | Anonymised placeholder for a dataset or benchmark. No public "IIS3D" dataset was found online. This is expected while the link is withheld for review. Make sure the camera-ready version gives the real citation. | none (cannot verify) | none |
| bergna2024lgnsde | FIX | Wrong title, incomplete author list ("and others"), and a stale venue. arXiv:2408.16115 is titled "Uncertainty Modeling in Graph Neural Networks via Stochastic Differential Equations"; LGNSDE is the name of the model, not the title. It has five authors and was published at ICLR 2025 (Spotlight). | author = {Bergna, Richard and Calvo-Ordo{\~n}ez, Sergio and Opolka, Felix L. and Li{\`o}, Pietro and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel}; title = {Uncertainty modeling in graph neural networks via stochastic differential equations}; booktitle = {International Conference on Learning Representations (ICLR)}; year = {2025}; (entry type @inproceedings; remove journal) | https://proceedings.iclr.cc/paper_files/paper/2025/hash/62b4fea131cfd5b7504eae356b75bbd8-Abstract-Conference.html ; https://arxiv.org/abs/2408.16115v5 |
| clopper1934 | OK | None. Authors, title, Biometrika 26(4):404–413 and the year 1934 all match. Optionally add doi = {10.1093/biomet/26.4.404}; the JSTOR DOI is 10.2307/2331986. | none (optional: doi = {10.1093/biomet/26.4.404}) | https://www.scilit.net/publications/db65b7b21348fddd8a4297389dc98cf3 |
| dejorge2022nfgsm | OK | None. The title, all 7 authors in order, NeurIPS and 2022 match. NeurIPS lists the first author as "Pau de Jorge Aranda"; "de Jorge, Pau" matches the arXiv version. Optionally add volume = {35}, pages = {12881--12893}. | none (optional: volume = {35}, pages = {12881--12893}) | https://proceedings.neurips.cc/paper_files/paper/2022/hash/5434a6b40f8f65488e722bc33d796c8b-Abstract.html |
| guo2017calibration | OK | None. Authors, title, ICML and 2017 match. Optionally add series = {PMLR}, volume = {70}, pages = {1321--1330}. | none (optional: volume = {70}, pages = {1321--1330}) | https://proceedings.mlr.press/v70/guo17a.html |
| jeong2020consistency | OK | None. Authors, title, NeurIPS 2020 (vol. 33) and year match. | none | https://papers.nips.cc/paper/2020/hash/77330e1330ae2b086e5bfcae50d9ffae-Abstract.html |
| lin2017focal | OK | None. Authors in order, title, ICCV and 2017 match. Optionally add pages = {2980--2988}, doi = {10.1109/ICCV.2017.324}. | none (optional: pages = {2980--2988}) | https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html |
| maurer2009bernstein | OK | None. Authors, title, COLT 2009 match. The COLT listing hyphenates "Sample-Variance"; arXiv and the PDF use "Sample Variance". Either is fine. | none | https://mlanthology.org/colt/2009/maurer2009colt-empirical ; https://arxiv.org/abs/0907.3740 |
| oh2024stablesde | OK | None. Authors, title, ICLR 2024 (Spotlight) match. The ICLR page spells the second author "Dongyoung Lim"; the paper PDF uses "Dong-Young Lim". Either is acceptable. | none | https://proceedings.iclr.cc/paper_files/paper/2024/hash/a61023ce36d21010f1423304f8ec49af-Abstract-Conference.html |
| salman2019smoothadv | OK | None. The author order matches the NeurIPS 2019 proceedings: Salman, Li, Razenshteyn, P. Zhang, H. Zhang, Bubeck, Yang. arXiv and MSR use a different order, so this order is right for the NeurIPS citation. | none | https://proceedings.neurips.cc/paper/2019/hash/3a24b25a7b092a252166a1641ae953e7-Abstract.html |
| tzen2019nsde | OK | None. Authors, title, arXiv:1905.09883 and 2019 match. No peer-reviewed venue was found, so the arXiv citation is appropriate. | none | https://arxiv.org/abs/1905.09883 |
| yang2020smoothing | OK | None. All 6 authors in order, the title, ICML 2020, PMLR 119 and pp. 10693–10705 match. | none | https://proceedings.mlr.press/v119/yang20c.html |

## Summary

- OK: 12
- FIX: 1 (bergna2024lgnsde)
- NOT FOUND: 0
- UNVERIFIABLE: 3 (anaedevha2026estc, anaedevha2026mstgnn, anon_iis3d)

Before submitting, open these DOIs by hand, because they could not be resolved from this session: 10.1109/EDM69524.2026.11632205 and 10.36227/techrxiv.176836457.72974723/v1.

## Corrected BibTeX for FIX rows

```bibtex
@inproceedings{bergna2024lgnsde,
  author    = {Bergna, Richard and Calvo-Ordo{\~n}ez, Sergio and Opolka, Felix L. and Li{\`o}, Pietro and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
  title     = {Uncertainty modeling in graph neural networks via stochastic differential equations},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```

The cite key `bergna2024lgnsde` is unchanged so that no `\cite` in the paper breaks. If you rename it to `bergna2025...`, update the citations as well. If you would rather cite the arXiv preprint, use: `journal = {arXiv preprint arXiv:2408.16115}`, `year = {2024}`, with the corrected title and authors above.
