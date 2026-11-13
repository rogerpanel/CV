# Adversarial-training modules for raising worst-case robustness of SODE-Guard (neural-SDE NIDS classifier, ICS3D, l-inf eps=0.03 z-space)

Research-environment note: in this session WebFetch could not reach arxiv.org, openreview.net, proceedings.neurips.cc, openaccess.thecvf.com or orbilu.uni.lu (DNS blocked by the sandbox network policy). All findings below come from search-result snippets of primary sources (papers, PMLR, GitHub, NeurIPS pages). Where a number came from a secondary summary, that is flagged. Exact tables (TRADES beta sweeps, CFA per-class numbers, TabularBench full leaderboard) should be checked against the PDFs before being quoted in a paper. To fetch the full texts, those hosts have to be allowed in the session's network settings.

## 1. Which AT variants (TRADES, MART, AWP, Fast-AT/FGSM-RS/GradAlign/N-FGSM, curriculum, SWA/EMA, label smoothing) beat PGD-AT, by how much, and which are cheapest for an SDE solver?

### Takeaway
Against an AutoAttack-style evaluation the reliable gains over vanilla PGD-AT on image benchmarks are small, a few points. TRADES gives about +1 pt AA, AWP about +2.5 to +3 pts, and early stopping or weight averaging recovers most of what fancier methods claim (Rice et al.). For an SDE, where every attack step costs a full forward and backward solve, the best value is probably N-FGSM or FGSM-RS single-step AT (about 1/3 to 1/10 the cost of PGD-10), plus AWP (about +8% cost), weight averaging (EMA/SWA, nearly free) and early stopping on robust validation. This is an inference from cost structure; no paper measures these on SDEs.

### Cited Findings
- AWP (Wu, Xia, Wang, NeurIPS 2020), WRN-34-10 CIFAR-10 l-inf 8/255, no extra data: clean 85.36%, AutoAttack 56.17% in the original AutoAttack leaderboard. For comparison, the Zhang et al. 2020b row shows clean 84.52%, PGD 54.36%, AA 53.51%. — [fra31/auto-attack README](https://github.com/fra31/auto-attack)
- A secondary summary of Wu et al. 2020 says AWP raises PreAct ResNet-18 robust test accuracy from 52.79% to 55.39% (not verified against the original). — [emergentmind AWP topic](https://api.emergentmind.com/topics/adversarial-weight-perturbation)
- Robust Weight Perturbation (RWP, follow-up to AWP), CIFAR-10 l-inf, "best" checkpoint robust accuracy: vanilla AT 52.32%, AT-AWP 55.54%, AT-RWP 58.55%. With PGD-20 (not AA): TRADES-AWP 59.20%, TRADES-RWP 60.45%. — [RWP, arXiv 2205.14826](https://arxiv.org/pdf/2205.14826)
- Robust overfitting (Rice, Wong, Kolter, ICML 2020): "overfitting to the training set does in fact harm robust performance to a very large degree" on SVHN, CIFAR-10, CIFAR-100 and ImageNet. "The performance gains of virtually all recent algorithmic improvements upon adversarial training can be matched by simply using early stopping." No remedy tested in isolation (including regularizers and augmentation) significantly beat early stopping. Overfitting starts shortly after the learning-rate decay. — [PMLR v119 rice20a](http://proceedings.mlr.press/v119/rice20a/rice20a.pdf); [arXiv 2002.11569](https://arxiv.org/pdf/2002.11569)
- Weight averaging reduces sensitivity to the early-stopping epoch, but averaged weights are still increasingly affected by overfitting after the learning-rate change. Flatness methods (AWP, Chen et al. 2020) are cited as mitigations. — [Random Forgetting for Robust Generalization, arXiv 2402.11733](https://arxiv.org/pdf/2402.11733)
- Bag of Tricks for AT (Pang et al., ICLR 2021): AT is "more sensitive to these usually overlooked hyperparameters, compared to standard training". The paper studies early stopping, label smoothing, weight decay, batch size, BN mode and optimizer, and evaluates with AutoAttack. The exact per-trick numbers were not retrievable here. — [ICLR 2021 slides](https://iclr.cc/media/iclr-2021/Slides/2754.pdf); [arXiv 2010.00467](https://arxiv.org/pdf/2010.00467); [code](https://github.com/P2333/Bag-of-Tricks-for-AT)
- N-FGSM (de Jorge et al., NeurIPS 2022, "Make Some Noise"): single-step AT that avoids catastrophic overfitting and "matches, or even surpasses, the robust accuracy of the regularized FGSM (GradAlign), while providing a 3x speed-up." — [NeurIPS 2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/5434a6b40f8f65488e722bc33d796c8b-Paper-Conference.pdf)
- FGSM-RS catastrophic overfitting depends on step size. CO happens when alpha/eps > 1 at eps=8/255, and when alpha/eps > 0.5 at eps=16/255. At 16/255, FGSM-RS reaches 16.73% robust accuracy with alpha=0.75·eps but suffers CO at 0.875·eps. — [Mitigating and Explaining CO in FGSM AT, arXiv 2103.15476](https://arxiv.org/pdf/2103.15476)
- GradAlign (Andriushchenko and Flammarion, NeurIPS 2020) prevents CO where FGSM and FGSM-RS fail, but there is "still a 5% gap to PGD-2 training for eps=8/255". With lambda=0.1, GradAlign suffers CO at eps=16/255, so lambda must grow with eps. — [Understanding and Improving Fast AT, arXiv 2007.02617](https://arxiv.org/pdf/2007.02617); [OpenReview "PGD-2 can be better than FGSM + GradAlign"](https://openreview.net/pdf/2b6182508bcb3b449c83c8ed234052400d217667.pdf)
- Conflicting evidence on N-FGSM at large eps: one evaluation reports N-FGSM suffers CO for eps > 16/255 and robust overfitting in [8/255, 16/255]. This contradicts the original paper's no-CO claim and points to dependence on the setup. — [search summary of ICLR 2022 submission](https://openreview.net/pdf?id=lifRwnIuAv0) (attribution of this specific claim to that exact PDF is uncertain)
- Also on the cost and stability of single-step AT: [Bag of Tricks for FGSM AT, arXiv 2209.02684](https://arxiv.org/pdf/2209.02684); [Efficient local linearity regularization (ELLE)](https://wrap.warwick.ac.uk/id/eprint/184678/1/WRAP-Efficient-local-linearity-regularization-overcome-catastrophic-overfitting-24.pdf)
- RobustBench is the maintained leaderboard. A 2026 snapshot lists top entries such as Wang2023Better WRN-70-16 at 93.25% clean and 70.69% AA, which rely on large models and diffusion-generated data. — [RobustBench paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/a3c65c2974270fd093ee8a9bf8ae7d0b-Paper-round2.pdf); [Auto-ART 2026 survey, arXiv 2604.20704](https://arxiv.org/pdf/2604.20704)

### Inferences
- **Cost per step for an SDE.** Each inner attack step needs one forward SDE solve plus one adjoint/backprop solve, so the costs scale as follows:
  - PGD-K-AT costs about (K+1) solve-pairs per minibatch.
  - TRADES with K steps costs about the same as PGD-K, plus a KL term on clean logits.
  - MART costs the same as PGD-K.
  - AWP adds about one extra solve-pair (one weight-ascent step). The original paper reports about 8% overhead on CNNs; this was not verified in this session.
  - FGSM-RS and N-FGSM cost 2 solve-pairs.
  - GradAlign needs double backprop through the solver, which is expensive and memory-heavy for SDE adjoints. It is probably the worst fit.
  - EMA/SWA and label smoothing are essentially free.

  The cheapest strong recipe is therefore N-FGSM (or PGD-2/3) + AWP + EMA + robust-val early stopping, with full PGD-10 or TRADES fine-tuning for the last few epochs if the budget allows.
- **Expected gains over PGD-AT.** On images, TRADES and AWP give +1 to +3 pts AA over PGD-AT. SODE-Guard (0.225) is 16 pts macro-F1 below PGD-AT MLP (0.383), so these modules alone are unlikely to close the gap unless the SDE is first trained with any AT. The SDE is possibly not adversarially trained at all, or is relying on stochastic/obfuscated gradients (Section 4). Step one is plain PGD-AT on the SDE, evaluated with EOT; TRADES/AWP/EMA come after.
- **Using TRADES or a smooth loss with a stochastic model.** A smooth KL-type robust loss (TRADES) may interact better with a stochastic model than hard CE on argmax, because the KL is computed on averaged logits. This is speculative.
- **Robust overfitting on 70k tabular samples.** It is likely severe, because tabular datasets are small and AT overfits quickly. Early stopping on worst-case macro-F1 measured on a held-out robust validation split (PGD-20 + EOT) is the minimum safeguard.

### Gaps
- Exact TRADES and MART RobustBench AA numbers and the beta ablation were not retrievable in this session. From prior knowledge, not verified here: TRADES used beta=6 for CIFAR-10 at eps=8/255, PGD-10 steps of 2/255 (0.007), WRN-34-10, giving about 53.1% AA. MART's improvements over TRADES under AA without extra data are small.
- Pang et al.'s label-smoothing and weight-decay findings could not be retrieved. From prior knowledge: weight decay 5e-4 is important, and label smoothing gives a small gain that can partly be gradient masking. Verify both.
- Curriculum and epsilon warm-up: no source retrieved. Linear eps ramp-up over the first ~10-20% of epochs is common practice in certified and fast AT, but no number was confirmed here.
- No paper found that reports TRADES, MART, AWP or N-FGSM gains specifically on tabular or NIDS data under AutoAttack.

## 2. Adversarial training under class imbalance: robust-accuracy disparity, and how to keep rare classes from collapsing

### Takeaway
AT amplifies class disparity. On CIFAR-10, the PGD-AT robust error is 82.4% for the "cat" class against a 56.4% average. Remedies that work: class-wise reweighting and re-margining (FRL), class-wise calibrated eps and beta plus fairness-aware weight averaging (CFA, CVPR 2023), and long-tail-specific rebalancing (RoBal, REAT, AT-BSL). For a 9-class imbalanced NIDS set evaluated by macro-F1, class-wise eps/beta and class-balanced or logit-adjusted robust loss are directly relevant.

### Cited Findings
- Xu, Liu, Li, Jain, Tang, "To be Robust or to be Fair" (ICML 2021). A PGD-AT ResNet-18 on CIFAR-10 has 93% clean and 67% PGD l-inf-8 robust accuracy on "automobile", but only 65% and 17% on "cat". The cat class has 32.8% standard error and 82.4% robust error, against model averages of 15.5% and 56.4%. — [PMLR v139 xu21b](http://proceedings.mlr.press/v139/xu21b/xu21b.pdf); [arXiv 2010.06121](https://arxiv.org/pdf/2010.06121)
- FRL (Fair Robust Learning) targets boundary-error and robust-error disparity. "FRL (Reweight) can achieve the minimal worst-class standard error and FRL (Reweight+Remargin) gives the minimum worst-class robust error." — [arXiv 2010.06121v2](https://arxiv.org/html/2010.06121v2)
- CFA (Wei, Wang, Guo, Wang, CVPR 2023, pp. 8193-8201) studies how classes respond differently to the perturbation margin, the regularization strength and weight averaging. It automatically customizes per-class configurations: Class-wise Calibrated Margin (CCM) for per-class eps, Class-wise Calibrated Regularization (CCR) for per-class TRADES beta, and Fairness-Aware Weight Averaging (FAWA). It reports improving "both overall robustness and fairness notably over other state-of-the-art methods". A secondary summary says FAT (instance-wise) reaches only 17.2% worst-class robust accuracy, below TRADES. — [CVPR 2023 page](https://openaccess.thecvf.com/content/CVPR2023/html/Wei_CFA_Class-Wise_Calibrated_Fair_Adversarial_Training_CVPR_2023_paper.html); [arXiv 2303.14460](https://arxiv.org/abs/2303.14460v1); [Liner summary (secondary)](https://liner.com/review/cfa-classwise-calibrated-fair-adversarial-training)
- Long-tailed AT, RoBal (Wu, Liu et al., "Adversarial Robustness under Long-Tailed Distribution"): a scale-invariant (cosine) classifier plus data re-balancing via margin engineering at training time and boundary adjustment at inference. — [Semantic Scholar](https://www.semanticscholar.org/paper/Adversarial-Robustness-under-Long-Tailed-Wu-Liu/31d5494ad29e93dc22a00965e7bbc21c77e7634f)
- REAT (Li et al. 2023, "Alleviating the Effect of Data Imbalance on Adversarial Training") combines RBL, a re-balancing loss that makes the model generate balanced adversarial examples, with TAIL, a regularizer that gives tail classes larger feature space. It claims higher clean and robust accuracy than RoBal. Note: REAT is by Li et al. 2023, not "Wang, ICDM 2022" as sometimes cited. — [arXiv 2307.10205](https://arxiv.org/html/2307.10205)
- Head-to-head comparisons conflict. In one comparison REAT beats RoBal on clean accuracy (74.52% vs 73.18%) but is lower on PGD and AA robust accuracy. In another table (CGRm authors), REAT trails RoBal on clean accuracy in several settings. — [CGRm, arXiv 2608.09688](https://arxiv.org/html/2608.09688); [AT-BSL / long-tailed self-distillation, arXiv 2503.06461](https://arxiv.org/pdf/2503.06461)
- AT-BSL (Yue et al. 2024) matches RoBal while significantly reducing training time and memory. Other follow-ups include TAET (two-stage adversarial equalization) and hard-example mining for robust fairness. — [arXiv 2503.06461](https://arxiv.org/pdf/2503.06461); [TAET arXiv 2503.01924](https://arxiv.org/pdf/2503.01924); [HAM arXiv 2308.01823](https://arxiv.org/pdf/2308.01823); [CODAT arXiv 2501.04527](https://arxiv.org/pdf/2501.04527)
- NIDS meta-analysis: UNSW-NB15 and CIC-IDS2017 account for 71% of dataset usage, and only 34% of studies report per-class recall for minority attack types. — [search snippet, Dataset-Calibrated Perturbation Budgets / related survey](https://doi.org/10.3390/technologies14100647) (exact attribution of the meta-analysis statistic is uncertain)

### Inferences
- **What to do for SODE-Guard (9 imbalanced classes, macro-F1 metric):**
  - Use class-balanced or logit-adjusted CE (or RoBal-style cosine classifier + margin) inside the AT loss, both for the outer loss and for generating balanced adversarial examples (REAT's RBL idea).
  - Use per-class eps/beta (CFA CCM/CCR). Rare classes, which sit closer to the boundary, get smaller training eps or smaller beta so they are not pushed into collapse.
  - Use FAWA-style EMA, keeping checkpoints only when worst-class robust recall does not drop.
  - Early stop on robust macro-F1, not on robust accuracy.
- **Weighted sampling of adversarial examples.** Oversampling rare classes during AT, as opposed to post-hoc, keeps the adversarial-example distribution balanced. Of the remedies above, this one is cheapest in solver cost.

### Gaps
- Exact CFA numbers could not be retrieved (CIFAR-10 average vs worst-class AA robust accuracy for PGD-AT, TRADES, FRL and CFA). From prior knowledge, unverified: CFA raises worst-class AA robust accuracy by a few points over TRADES at a similar average.
- No published study found that applies FRL, CFA or REAT to NIDS or tabular data.

## 3. Tabular-specific adversarial robustness: constrained attacks (CAA, MoEvA2, CPGD), TabularBench rankings, transfer of AT, NIDS datasets

### Takeaway
TabularBench (NeurIPS 2024 D&B) evaluates 200+ models on 5 real datasets, including the CTU-13 botnet dataset, under CAA. CAA combines CAPGD gradient search with MoEvA2 genetic search and is the strongest attack in 17 of 20 settings. Madry-style AT with CAPGD is the main robustifier. AT + data augmentation can beat AT alone, but the augmenter matters: CTGAN can catastrophically hurt robustness on CTU-13. AT alone "is not always enough", architecture matters (TabNet and TabTransformer vary widely), and some apparently robust models are degenerate single-class predictors. On NIDS data, AT gives modest gains under realistic constraints, and unconstrained l-p attacks overstate the threat.

### Cited Findings
- TabularBench (Simonetto, Ghamizi, Cordy, NeurIPS 2024 Datasets and Benchmarks Track): 7 robustification mechanisms inspired by CV defenses, 200+ models, five use-cases in finance, healthcare and security. Robustness is measured with the Constrained Adaptive Attack (CAA). — [NeurIPS 2024 abstract](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8f5defaa8dc728c3e4aaee59ac8ca942-Abstract.html); [papers.nips.cc D&B](https://papers.nips.cc/paper_files/paper/2024/hash/8f5defaa8dc728c3e4aaee59ac8ca942-Abstract-Datasets_and_Benchmarks_Track.html); [arXiv 2408.07579](https://arxiv.org/html/2408.07579v1); [leaderboard site](https://serval-uni-lu.github.io/tabularbench/)
- CAA (Simonetto et al. 2024) combines gradient-based (CAPGD) and search-based (MoEvA2) attacks and outperforms all existing attacks in 17 of the 20 settings studied. — [arXiv 2406.00775](https://arxiv.org/pdf/2406.00775); [TabularBench](https://arxiv.org/pdf/2408.07579)
- Augmentations evaluated in TabularBench: WGAN, TableGAN, CTGAN, TVAE and a tabular CutMix, each combined with or without AT. — [Moonlight review (secondary)](https://www.themoonlight.io/en/review/tabularbench-benchmarking-adversarial-robustness-for-tabular-deep-learning-in-real-world-use-cases); [TabularBench PDF via OpenReview](https://openreview.net/notes/edits/attachment?id=mxGvpnraHB&name=pdf)
- CTU-13 (botnet) table excerpts, read as robust accuracy under CAA:
  - TabTransformer, AT, no augmentation: about 0.967. With CTGAN augmentation: about 0.017.
  - VIME, AT, no augmentation: about 0.974. With CTGAN: about 0.000. Other augmenters keep it near 0.97.
  - RLN, AT + CTGAN: about 0.561. With WGAN or CutMix: near 0.95-1.0.

  These numbers come from search excerpts of table rows; the column semantics must be verified. — [TabularBench PDF](https://orbilu.uni.lu/bitstream/10993/63141/1/tabular_bench_benchmarking_adversarial_robustness_for_tabular_deep_learning_in_real_world_use_cases.pdf)
- With standard training, in-distribution performance on CTU and URL is unreliable as an indicator of robustness. All models have similar ID performance, but some augmentations yield robust models. With augmentation + AT, ID and robust performance are correlated. An author's slides state that AT with data augmentation outperforms AT alone, which conflicts with the CTGAN/CTU rows above, so it depends on the augmenter. — [TabularBench PDF](https://arxiv.org/pdf/2408.07579); [Simonetto slides](https://simonetto.lu/files/tabularbench_slides.pdf)
- Degenerate robustness: "some achieve this robustness solely by consistently predicting the '1' class." Report MCC or macro metrics, not accuracy. — [TabularBench PDF](https://orbilu.uni.lu/bitstream/10993/63141/1/tabular_bench_benchmarking_adversarial_robustness_for_tabular_deep_learning_in_real_world_use_cases.pdf)
- Secondary-summary numbers (unverified):
  - URL dataset: TabTransformer 93.9% ID but 56.7% robust with AT, up from 8.9% robust with standard training. TabNet 99.5% ID and 91.8% robust with AT.
  - WiDS: TabNet 98.4% ID but 58.4% robust with AT.
  - The authors conclude AT alone is not always enough.

  — [Liner review (secondary)](https://liner.com/review/tabularbench-benchmarking-adversarial-robustness-for-tabular-deep-learning-in-realworld)
- Insights on adversarial attacks for tabular ML (2025) is a follow-up from the same line of work. — [arXiv 2506.15506](https://arxiv.org/pdf/2506.15506)
- UNSW-NB15:
  - An unconstrained Carlini-Wagner attack reduced a detector to 0.38 accuracy. Under realistic feature constraints the same attack reduced it only to 0.748.
  - PGD-based AT raised robust accuracy from 0.818 to 0.859, with under 1 pt clean loss.

  — [Evaluating and Improving Adversarial Robustness of ML-Based NIDS (ResearchGate)](https://www.researchgate.net/publication/352215283_Evaluating_and_Improving_Adversarial_Robustness_of_Machine_Learning-Based_Network_Intrusion_Detectors) (attribution from snippet; verify)
- Another UNSW-NB15 adversarially trained NN reports 86.04% clean and over 80% under PGD and FGSM (preprint-level). — [ResearchGate 394893640](https://www.researchgate.net/publication/394893640_Enhancing_Adversarial_Robustness_in_Network_Intrusion_Detection_A_Novel_Adversarially_Trained_Neural_Network_Approach)
- A2PM (Vitorino et al., "Adaptative Perturbation Patterns: Realistic Adversarial Learning for Robust Intrusion Detection") defines constraint levels for realistic NIDS examples. It is evaluated on CIC-IDS2017 and IoT-23 under a gray-box setting, and its examples are used for adversarial training. — [arXiv 2203.04234](https://arxiv.org/pdf/2203.04234)
- Fixed feature-space perturbation bounds are not comparable across NIDS datasets. Measured constrained evasion cost differs by more than two orders of magnitude across CICIDS2017, TON_IoT and UNSW-NB15, and the authors recommend calibrating eps per dataset. — [Dataset-Calibrated Perturbation Budgets, Technologies 14(10):647](https://doi.org/10.3390/technologies14100647)
- Survey of adversarial challenges in NIDS. — [arXiv 2409.18736](https://arxiv.org/pdf/2409.18736)

### Inferences
- **Choice of attack for training.** For SODE-Guard, AT should use the attack it will be evaluated with. If CAA-style constrained attacks will be reported, train with CAPGD (constraint-repairing PGD) rather than plain l-inf PGD. TabularBench's main defense is exactly Madry-AT with CAPGD.
- **eps=0.03 in z-score space.** This is a very small budget, about 3% of one standard deviation per feature. That the MLP's PGD-AT robust macro-F1 (0.383) is so close to its clean score (0.409) is consistent with a small budget. The large SODE-Guard drop (to 0.225) therefore indicates a lack of local smoothness, not an intrinsically hard threat model.
- **Augmentation.** If used with AT (CutMix or WGAN were safest on CTU-13 in TabularBench), validate under the robust metric, because CTGAN destroyed robustness for several architectures on CTU-13.
- **Metrics.** Always report MCC or macro-F1 plus per-class robust recall, to rule out degenerate "always-attack" or "always-benign" robustness.

### Gaps
- Could not retrieve TabularBench's full ranking table (which architecture is best overall under CAA after AT), its exact AT hyperparameters (CAPGD steps, eps, epochs), or whether AT with CAPGD transfers to MoEvA2. Verify in the PDF.
- No retrieved results for TabularBench-style evaluation on CIC-IDS2017 or UNSW-NB15. TabularBench's security datasets are CTU-13, URL and Malware.
- Apruzzese et al. ("Modeling Realistic Adversarial Attacks against NIDS", DTRAP 2022) and Sheatsley et al. (constrained domains, S&P/IEEE) were not retrieved in this session.

## 4. Adversarial training of neural ODEs/SDEs specifically: does AT help them more or less than MLPs? AutoAttack evaluations

### Takeaway
Most ODE/SDE robustness claims come without AT or with weak, PGD-only evaluation, and several were shown to rest on obfuscated gradients from adaptive solvers or stochasticity. The best-documented AT + ODE result is SODEF (NeurIPS 2021). On a TRADES backbone it reported AutoAttack l-inf 53.69% for TRADES vs 57.76% for TRADES+SODEF on CIFAR-10. Croce et al. later showed SODEF's AutoAttack evaluation was largely transfer-based and unreliable. No paper was found showing that AT helps ODEs/SDEs more than MLPs under a fully adaptive evaluation. SODE-Guard's 0.225 vs 0.383 result fits the pattern that, once gradients are honest, continuous-depth models without AT are not intrinsically robust.

### Cited Findings
- Neural SDE (Liu, Xiao, Si, Cao, Kumar, Hsieh, 2019) injects noise (including dropout-like noise) into neural ODE dynamics. It claims better generalization and resistance to adversarial and non-adversarial perturbations, with theoretical analysis. A secondary summary says gains are significant against L2 PGD, with dropout-style noise strongest. No adversarial-training procedure was confirmed. — [arXiv 1906.02355](https://arxiv.org/abs/1906.02355); [DeepAI summary](https://deepai.org/publication/neural-sde-stabilizing-neural-ode-networks-with-stochastic-noise)
- Yan et al. (ICLR 2020), "On Robustness of Neural ODEs": ODENets are more robust than CNNs to Gaussian noise and adversarial examples. TisODE (time-invariant steady ODE) can work together with neural SDE to further boost robustness. — [arXiv 1910.05513](https://arxiv.org/pdf/1910.05513); [Semantic Scholar](https://www.semanticscholar.org/paper/On-Robustness-of-Neural-Ordinary-Differential-Yan-Du/00d4f1c5f11ecbbc1a033e6675934703f09016f8)
- Huang et al. (PMLR v145, 2022), "Adversarial Robustness of Stabilized Neural ODE Might be from Obfuscated Gradients". SONet (a skew-symmetric, Lyapunov-stable neural ODE) looks robust without AT. The robustness is attributed to gradient masking by the adaptive DOPRI5 solver: it defeats PGD but not CW or the gradient-free SPSA. They warn of a "false sense of adversarial robustness" under PGD. — [arXiv 2009.13145](https://arxiv.org/abs/2009.13145); [PMLR](https://proceedings.mlr.press/v145/huang22a/huang22a.pdf)
- A counterpoint paper argues neural ODE robustness comes from the integration time horizon rather than obfuscated gradients, and proposes ResNet-teacher knowledge distillation. Longer horizons improve robustness. — [arXiv 2203.05103](https://arxiv.org/pdf/2203.05103)
- SODEF (Kang, Song, Ding, Tay, NeurIPS 2021) appends a neural ODE layer after a feature extractor, forces features toward Lyapunov-stable equilibria, and fixes a diverse final FC layer (unit-norm rows with minimized cosine similarity). Reported AutoAttack on CIFAR-10: TRADES l-inf 53.69 vs TRADES+SODEF 57.76, and TRADES l2 59.42 vs TRADES+SODEF 67.75. — [NeurIPS 2021 PDF](https://proceedings.neurips.cc/paper/2021/file/7d5430cf85f78c4b7aa09813b14bce0d-Paper.pdf); [code](https://github.com/kangqiyu/SODEF)
- Critique (Croce et al., "Evaluating the Adversarial Robustness of Adaptive Test-time Defenses", ICML 2022): AutoAttack by default returns the original image when no adversarial example is found, so transferred points were mostly clean images. SODEF was evaluated by transferring APGD from the static TRADES model rather than attacking SODEF end-to-end, which likely overstates robustness. — [arXiv 2202.13711](https://arxiv.org/pdf/2202.13711)
- Recent work: "Learning Aligned Stability in Neural ODEs: Reconciling Accuracy with Robustness" (2025) and "Improving the robustness of neural ODEs with minimal weight perturbation" (2025). — [arXiv 2509.21879](https://arxiv.org/pdf/2509.21879); [arXiv 2501.10740](https://arxiv.org/html/2501.10740)

### Inferences
- **Ranking of modules for SODE-Guard, by expected impact:**
  1. **Train the SDE with honest AT.** Generate attacks with the same EOT used at evaluation, for example PGD-K over the averaged logits of M SDE samples, or one sample per step with fixed solver settings. If training attacks see a single noisy path while the evaluation uses EOT, the training adversary is weaker than the evaluator, and the robustness learned is mostly from gradient noise.
  2. **Put the backbone in the AT loop.** If the encoder or feature extractor is AT-trained and the SDE block is appended (SODEF style), the SODEF numbers suggest +4 pts AA over TRADES. Given the Croce critique, though, only an end-to-end adaptive evaluation counts.
  3. **Use a fixed-step solver during AT** (Euler-Maruyama with few steps). This keeps the cost per attack step bounded and avoids adaptive-solver gradient masking, which honest evaluations would otherwise undo.
  4. **Add Lyapunov/stability or Jacobian-norm regularizers** (SODEF-style, or the drift Jacobian Frobenius norm) as cheap complements to AT. Their stand-alone robustness should not be trusted.
- **Does AT help ODEs/SDEs more or less than MLPs?** There is no direct evidence. The only AT + ODE AutoAttack evidence (SODEF) shows a small gain over the same TRADES backbone, which suggests "slightly more" at best, and that is contested.
- **Main SDE-specific cost.** Each PGD step needs one SDE forward and one backward pass, and EOT multiplies that by M. Single-step N-FGSM with M=1-4 samples, plus a short final phase of multi-step TRADES, is the cost-feasible path.

### Gaps
- No paper found that trains a neural SDE classifier with PGD-AT or TRADES and evaluates it with full adaptive AutoAttack + EOT.
- No ODE/SDE AT studies on tabular or NIDS data were found.

## 5. Typical hyperparameters and pitfalls (eps schedule, TRADES beta, steps, step size, catastrophic and robust overfitting, early stopping)

### Takeaway
Standard image settings (eps 8/255, PGD-10 at step ~eps/4, TRADES beta=6, AWP gamma about 5e-3, weight decay 5e-4) transfer only loosely to tabular z-space. The main pitfalls are catastrophic overfitting in single-step AT (FGSM-RS step > eps at 8/255, or > 0.5·eps at 16/255), robust overfitting after learning-rate decay (fix with early stopping on robust validation, AWP, or weight averaging), gradient masking from stochastic or adaptive solvers, and degenerate single-class robustness on imbalanced tabular data.

### Cited Findings
- FGSM-RS step-size thresholds for catastrophic overfitting: alpha/eps > 1 at 8/255 and > 0.5 at 16/255. FGSM-RS can avoid CO at 8/255 on CIFAR-10, but both FGSM-RS and GradAlign hit CO at 8/255 on ImageNet. — [arXiv 2103.15476](https://arxiv.org/pdf/2103.15476); [Kang & Moosavi-Dezfooli, Understanding CO, arXiv 2105.02942](https://arxiv.org/pdf/2105.02942)
- N-FGSM avoids CO at about 3x less cost than GradAlign. Its behavior at large eps is disputed. — [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/5434a6b40f8f65488e722bc33d796c8b-Paper-Conference.pdf)
- Robust overfitting appears shortly after the learning-rate drop. Early stopping matches most algorithmic gains. — [Rice et al., ICML 2020](http://proceedings.mlr.press/v119/rice20a/rice20a.pdf); [supplement](http://proceedings.mlr.press/v119/rice20a/rice20a-supp.pdf)
- Weight averaging reduces sensitivity to the stopping epoch but does not eliminate robust overfitting. — [arXiv 2402.11733](https://arxiv.org/pdf/2402.11733)
- AT is more hyperparameter-sensitive than standard training (weight decay, label smoothing, BN mode, early stopping). — [Pang et al., ICLR 2021 slides](https://iclr.cc/media/iclr-2021/Slides/2754.pdf)
- Adaptive solvers can mask gradients (Huang et al.). Transfer-based AutoAttack on dynamical defenses overstates robustness (Croce et al.). — [arXiv 2009.13145](https://arxiv.org/abs/2009.13145); [arXiv 2202.13711](https://arxiv.org/pdf/2202.13711)
- Calibrate eps per NIDS dataset, because constrained evasion cost varies over 100x across CICIDS2017, TON_IoT and UNSW-NB15. — [Technologies 14(10):647](https://doi.org/10.3390/technologies14100647)

### Inferences
- **Suggested starting config for SODE-Guard** (eps=0.03 z-space, 70k flows, 9 classes):
  - **Training adversary:** PGD-5 to 10 with step size 0.0075 (eps/4) and a random start, or N-FGSM with noise k=2·eps and step = eps, without clipping to the eps-ball.
  - **Training eps:** about 1.0-1.25× the test eps. Ramp eps linearly from 0 over the first 10-20% of epochs; this matters for SDE stability.
  - **TRADES:** beta in {1, 3, 6}, tuned on robust validation macro-F1.
  - **AWP:** gamma about 5e-3, starting after warm-up.
  - **EMA:** decay 0.999.
  - **Class-balanced loss** with per-class eps/beta (CFA-style).
  - **Early stopping:** on PGD-20 + EOT robust macro-F1 from a held-out validation split.
  - **EOT samples:** M during training equal to or close to M at evaluation.
  - **Solver:** fixed-step solver with the same step count in training and evaluation.

  This is a synthesis, not a tested recipe.
- **Robust overfitting on small tabular data.** It probably appears within tens of epochs. Monitor the train-vs-validation robust gap every epoch.

### Gaps
- The TRADES beta default (6) and AWP gamma (5e-3) come from prior knowledge of the original papers and could not be re-verified in this session.
- No tabular-specific sweep of TRADES beta, step count or eps schedule was found.
