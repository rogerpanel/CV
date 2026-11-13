# ODE/SDE architecture modules for adversarial robustness: claimed vs. adaptively verified

Research-environment note: in this session, WebFetch could not reach arxiv.org, pdf.arxiv.org, ar5iv.org, proceedings.mlr.press, openaccess.thecvf.com or api.deepai.org (DNS blocked by the sandbox network policy). github.com was reachable. Most findings below therefore come from search-result abstracts and snippets, plus the SODEF GitHub README, which was fetched in full. I could not extract exact tables from the primary PDFs of Liu et al. CVPR 2020, TisODE, Huang et al. (SONet), or Croce et al. 2022. Those numbers are listed under Gaps rather than guessed.

## Q1. Neural SDE noise injection (Liu et al. CVPR 2020 / Neural SDE, SDE-Net, RSE, PNI): did adaptive evaluations break them?

### Takeaway
Noise-injection defences (Neural SDE, RSE, PNI) were evaluated mainly with non-EOT or weak white-box attacks. The re-evaluation literature holds that EOT gradient estimation is the right tool for noise inside the network, and it found PNI still vulnerable under EOT. No source found in this session shows a neural-SDE diffusion term adding robustness that survives EOT/AutoAttack on its own. This matches the SODE-Guard observation that learned diffusion adds nothing under EOT-PGD or an expected-margin attack.

### Cited Findings
- Liu, Xiao, Si, Cao, Kumar, Hsieh, "Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise" (arXiv 1906.02355, June 2019), published at CVPR 2020 as "How Does Noise Help Robustness? Explanation and Exploration under the Neural SDE Framework". The method adds a diffusion term so that dropout (Bernoulli), additive and multiplicative noise are modelled as SDEs. The authors explain the robustness with SDE stability theory and report that the Neural SDE generalises better than a Neural ODE and resists adversarial and non-adversarial perturbations better. Their motivation is that a neural ODE system "is often unstable---a small input perturbation will be amplified through the ODE system and could eventually blow up". — [arXiv 1906.02355](https://arxiv.org/abs/1906.02355v1); [CVPR 2020 open access](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_How_Does_Noise_Help_Robustness_Explanation_and_Exploration_under_the_CVPR_2020_paper.html); [Google Research page](https://research.google/pubs/how-does-noise-help-robustness-explanation-and-exploration-under-the-neural-sde-framework/)
- A third-party summary says the analysis shows perturbations "decay exponentially" in Neural SDEs under certain conditions. This was not confirmed against the primary text. — [Emergent Mind summary](https://emergentmind.com/papers/1906.02355)
- PNI (He, Rakin, Fan, CVPR 2019) injects trainable Gaussian noise into activations or weights at each layer, with the noise scale learned through min-max adversarial training. It claims gains against PGD, C&W and FGSM, white-box and black-box. — [CVPR 2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Parametric_Noise_Injection_Trainable_Randomness_to_Improve_Deep_Neural_Network_CVPR_2019_paper.pdf); [arXiv 1811.09310](https://www.arxiv.org/abs/1811.09310)
- "Gradient Obfuscation Checklist Test Gives a False Sense of Security" (arXiv 2206.01705) reports that PNI, "which does not exhibit any of the five characteristics" of the Athalye et al. obfuscated-gradient checklist, "is still vulnerable to attacks with the EoT gradient estimation". It also argues that robustness claims need "a significant improvement over [an ablated] baseline". It states that EOT gradient estimation "is usually more effective when dealing with noise inside the network". — [arXiv 2206.01705](https://arxiv.org/pdf/2206.01705v1.pdf); [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2206.01705)
- In PNI the noise-scale coefficient is a learnable parameter. Follow-up work (e.g., Learn2Perturb) adds "some noise-promoting regulariser to prevent the noise from shrinking to zero during training". — [arXiv 2206.01705](https://arxiv.org/pdf/2206.01705v1.pdf) (search-snippet level)
- "Attacking Adversarial Defences by Smoothing the Loss Landscape" (arXiv 2208.00862) is a further attack paper on stochastic defences, found by search. Its content was not extracted. — [arXiv 2208.00862](https://arxiv.org/pdf/2208.00862)

### Inferences
- The recurring pattern: noise-injection gains reported under single-sample PGD shrink or disappear once the attacker averages gradients over noise draws (EOT) or attacks the expected logit/margin. SODE-Guard's 256-path expected-margin attack is the appropriate adaptive attack, so its null result is consistent with the literature, not an anomaly.
- If PNI's adversarially trained noise still falls to EOT, a diffusion term g trained without adversarial training should not be expected to help. Any residual benefit from noise is likely to come through its regularising effect during (adversarial) training, not through test-time randomness.

### Gaps
- I could not retrieve exact numbers from Liu et al. CVPR 2020: dataset (CIFAR-10 / STL-10 / Tiny-ImageNet), attack steps, eps, whether attacks used EOT or noise averaging, and whether sigma is fixed or learned. To verify, unconfirmed recollection: sigma is a fixed hyperparameter in Neural SDE, noise is kept at test time with multi-sample averaging, and the robustness was evaluated mainly with non-EOT PGD/C&W.
- Kong et al. SDE-Net (ICML 2020) was not retrieved in this session. To my knowledge it targets uncertainty and OOD detection, not adversarial robustness, so it is unlikely to provide an adaptive robustness evaluation. Not verified here.
- RSE (Liu et al. ECCV 2018, random self-ensemble): the search returned no re-evaluation results. I could not confirm whether Athalye et al. 2018 or Tramer et al. 2020 attacked RSE or PNI specifically.
- Exact EOT robust-accuracy numbers for PNI from arXiv 2206.01705 could not be extracted.

## Q2. Steady-state / stability modules (TisODE, SODEF, SONet, LyaNet, Carrara, Jacobian/kinetic regularisation): which survived adaptive evaluation, and which were gradient masking?

### Takeaway
The strongest independent evidence is that early ODE robustness came from obfuscated gradients in adaptive solvers. Huang et al. (SONet) show that DOPRI5 masks gradients so that PGD fails, while CW and gradient-free SPSA are not fooled. Carrara et al. tie neural-ODE robustness to the solver tolerance. SODEF's AutoAttack gains over TRADES (CIFAR-10 L-inf 53.69 to 57.76%) are self-reported. Croce et al. (ICML 2022) found that transferred surrogate-loss attacks bring SODEF below its own TRADES base. TisODE and LyaNet have no independent adaptive re-evaluation that I could locate.

### Cited Findings
- **SONet / Huang, Yu, Zhang, Ma, Yao, "Adversarial Robustness of Stabilized NeuralODEs Might be from Obfuscated Gradients"** (arXiv 2009.13145; MSML 2022, PMLR v145). The ODE blocks are skew-symmetric and proved input-output stable. With natural training the model shows non-trivial white-box robustness. However, "the adaptive stepsize numerical ODE solver, DOPRI5, has a gradient masking effect that fails the PGD attacks ... but the same effect does not fool the CW attack or the gradient-free SPSA attack". The authors conclude that "the adversarial robustness of ODE-based networks mainly comes from the obfuscated gradients in numerical ODE solvers". — [PMLR v145](https://proceedings.mlr.press/v145/huang22a.html); [arXiv 2009.13145](https://arxiv.org/abs/2009.13145)
- **Carrara et al.** WIFS 2019 was the first study of neural ODE classifiers under adversarial attack. It reported them more robust than ResNets under PGD and Carlini-Wagner and tied this to the adaptive solver's tunable accuracy-efficiency trade-off. — [CNR preprint](https://openportal.isti.cnr.it/data/2021/454312/2021_454312.preprint.pdf); [IRIS CNR](https://iris.cnr.it/handle/20.500.14243/398281)
- Carrara et al., *Information* 13(12):555 (2022), "Improving the Adversarial Robustness of Neural ODE Image Classifiers by Tuning the Tolerance Parameter". Tuning the solver tolerance at prediction time "grants robustness against adversarial attacks". The paper includes a section on adaptive attackers who also control the attack-time tolerance. — [MDPI Information 13/12/555](https://mdpi.com/2078-2489/13/12/555)
- A 2021 Carrara et al. preprint detects adversarial examples by randomising the solver tolerance across a majority-voting ensemble. It reports high rejection rates while keeping most clean accuracy. — [CNR preprint](https://openportal.isti.cnr.it/data/2021/454312/2021_454312.preprint.pdf)
- **TisODE** (Yan, Du, Tan, Feng, ICLR 2020, "On Robustness of Neural Ordinary Differential Equations") removes time dependence from the dynamics (time-invariant f(z)). It adds a steady-state regulariser that penalises the integrated magnitude of the dynamics over a later time window, so trajectories settle. It claims to outperform vanilla neural ODEs and to combine with other robust architectures. Evaluation covered Gaussian noise, FGSM and PGD. — [arXiv 1910.05513](https://arxiv.org/pdf/1910.05513); [ICLR 2020 poster](https://iclr.cc/virtual/2020/poster/1507)
- **SODEF** (Kang, Song, Ding, Tay, NeurIPS 2021, "Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attacks"). Architecture: feature extractor, then a neural ODE layer, then one or more FC layers. Training pushes extracted features into neighbourhoods of Lyapunov-stable equilibrium points of the ODE, so a perturbed input converges to the same solution as the clean one. The paper reports PGD-accuracy gains over TisODE of roughly 44, 53 and 19 points on MNIST, CIFAR-10 and CIFAR-100. — [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/7d5430cf85f78c4b7aa09813b14bce0d-Abstract.html); [arXiv 2110.12976](https://arxiv.org/pdf/2110.12976); [ar5iv](https://ar5iv.labs.arxiv.org/html/2110.12976)
- SODEF dependencies include `torchdiffeq` and `geotorch` (geotorch is used for constrained/orthogonal parameterisations). The README does not state the solver, integration time or regulariser weights. — [GitHub KANGQIYU/SODEF](https://github.com/KANGQIYU/SODEF)
- **Croce et al., ICML 2022, "Evaluating the Adversarial Robustness of Adaptive Test-time Defenses"** (arXiv 2202.13711). The evaluation used CIFAR-10, L-inf eps=8/255, nine adaptive test-time defences. None "significantly improve upon static defenses"; some "even weaken the underlying static model while simultaneously increasing inference computation". The paper extends the Carlini et al. 2019 checklist with steps specific to adaptive defences. — [PMLR v162](https://proceedings.mlr.press/v162/croce22a.html); [arXiv 2202.13711](https://arxiv.org/pdf/2202.13711)
- Croce et al. on SODEF: by default AutoAttack "returns the original images when they are originally misclassified or when no adversarial perturbation is found". Because TRADES is highly robust, "most of the transferred points are clean images", which undermines SODEF's transfer evaluation. When they instead transferred inputs that maximise surrogate losses (APGD with several losses on the TRADES model), "the robustness of SODEF is lower than that of the original TRADES classifiers". — [arXiv 2202.13711](https://arxiv.org/pdf/2202.13711) (via search snippets)
- **LyaNet** (Rodriguez, Ames, Yue, ICML 2022, PMLR 162:18687-18703). A Lyapunov loss pushes the inference dynamics to converge quickly to the correct prediction. The authors argue that minimising it guarantees exponential convergence and a robustness guarantee. They report better accuracy, faster convergence and improved adversarial robustness over standard Neural ODE training. One variant avoids backpropagating through the solver or adjoint. Code: github.com/ivandariojr/LyapunovLearning. I found no independent re-evaluation. — [arXiv 2202.02526](https://arxiv.org/abs/2202.02526); [ICML 2022 poster](https://icml.cc/virtual/2022/poster/17059); [mlanthology](https://mlanthology.org/icml/2022/rodriguez2022icml-lyanet)
- **Lipschitz / Gronwall control:** arXiv 2410.10744 finds that controlling the Lipschitz constant of the ODE dynamics (motivated by Gronwall's inequality) can significantly improve robustness. Ortho-ODE ("Enhancing Robustness of Neural ODEs against Adversarial Attacks") is a related orthogonality-constrained approach. Neither was verified under adaptive attack in this session. — [arXiv 2410.10744](https://pdf.arxiv.org/abs/2410.10744); [Ortho-ODE (aitopics)](https://aitopics.org/doc/arxivorg:17A16E40)
- **AROS** (ICLR 2025) uses neural ODEs with Lyapunov stability for adversarially robust OOD detection on image benchmarks such as CIFAR-10. — [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/30255)

### Inferences
- SODE-Guard uses fixed-step Euler-Maruyama, so it largely avoids the adaptive-solver gradient masking that inflated SONet and Carrara numbers. That helps explain why its ODE/SDE variants match the MLP under strong attacks: there is no masking artefact to inflate them, and no true stability mechanism either.
- Of the stability modules, only SODEF has AutoAttack-level numbers, and those come on top of an adversarially trained extractor. Even there, the independent evaluation shows no gain over the static base. The defensible conclusion is that stability or equilibrium modules provide at most a few points over adversarial training, and that the claimed gains are fragile.
- Large SODEF gains under FAB-T (54.18 to 82.92%) versus small gains under AutoAttack (53.69 to 57.76%) look like attack-specific gradient issues. A large gap between attacks is a standard gradient-obfuscation warning sign (my reading of the SODEF README table).
- Implementation implication: when evaluating an ODE/SDE block, attack with (a) fixed-step solvers and full backprop, not adjoint or adaptive tolerances, (b) gradient-free attacks (SPSA/Square), and (c) transfer from the base model using surrogate-loss maximisers, not AutoAttack's clean-returning outputs.

### Gaps
- Exact Huang et al. SONet numbers (FGSM, PGD, CW, SPSA accuracies; MNIST/CIFAR-10; DOPRI5 vs fixed-step) could not be extracted (PMLR and arXiv blocked).
- Exact TisODE numbers could not be extracted. To verify: TisODE reports MNIST/SVHN/CIFAR-10 accuracies under Gaussian noise, FGSM and PGD, with adaptive solvers and without adversarial training.
- Exact Croce et al. SODEF robust-accuracy numbers after re-evaluation were not extracted. The snippets confirm only "lower than the original TRADES classifier". The base TRADES AutoAttack accuracy in SODEF's own setup is 53.69% L-inf.
- "Defending against adversarial attacks via neural dynamical systems" (Li et al., NeurIPS 2022, ASODE?), Hoffman et al. 2019 Jacobian regularisation, and Finlay et al. 2020 ("How to train your neural ODE": kinetic energy plus Jacobian Frobenius regularisation) were not retrieved. To my knowledge Finlay et al. targets training speed and regularity, not adversarial robustness. Hoffman et al. reports robustness gains under PGD on MNIST/CIFAR without AutoAttack-level adaptive evaluation. Both are unverified here.
- Tramer et al. 2020 ("On Adaptive Attacks to Adversarial Example Defenses") was not retrieved. I could not confirm whether it includes any ODE/SDE or noise-injection defence.

## Q3. Combining ODE/SDE modules with adversarial training: reported AutoAttack numbers

### Takeaway
The only concrete AutoAttack numbers found come from SODEF on frozen adversarially trained extractors. On CIFAR-10 they are +4.07 points L-inf over TRADES and +4.70 points L-inf over Rebuffi2021. These are self-reported, and Croce et al. found that surrogate-loss transfer attacks bring SODEF below TRADES.

### Cited Findings
- SODEF README, CIFAR-10, TRADES base (frozen extractor), L-inf eps=8/255: Clean 85.48 to 85.18; APGD-CE 56.08 to 70.90; APGD-DLR-T 53.70 to 64.15; FAB-T 54.18 to 82.92; Square 59.12 to 62.21; **AutoAttack 53.69 to 57.76**. — [GitHub KANGQIYU/SODEF](https://github.com/KANGQIYU/SODEF)
- Same, L2 eps=0.5: APGD-CE 61.74 to 74.35; APGD-DLR-T 59.22 to 68.55; FAB-T 60.31 to 83.15; Square 72.65 to 76.02; **AutoAttack 59.42 to 67.75**. — [GitHub KANGQIYU/SODEF](https://github.com/KANGQIYU/SODEF)
- Transfer attack (AA L-inf 8/255, generated on the original pretrained model) on TRADES+SODEF: 61.94%. — [GitHub KANGQIYU/SODEF](https://github.com/KANGQIYU/SODEF)
- Rebuffi2021 base (RobustBench `Rebuffi2021Fixing_70_16_cutmix_extra`, frozen except the final FC), L-inf 8/255: Clean 92.23 to 93.73; **AutoAttack 66.58 to 71.28**; transfer 73.38. Per-attack breakdown "will report ... soon" (never added as of this fetch). — [GitHub KANGQIYU/SODEF](https://github.com/KANGQIYU/SODEF)
- The README does not mention the Croce et al. critique or any updated evaluation. — [GitHub KANGQIYU/SODEF](https://github.com/KANGQIYU/SODEF)
- Croce et al.: surrogate-loss transfer from TRADES leaves SODEF less robust than TRADES itself, and adaptive test-time defences in general did not significantly beat static defences. — [arXiv 2202.13711](https://arxiv.org/pdf/2202.13711); [PMLR](https://proceedings.mlr.press/v162/croce22a.html)
- Huang et al. also study adversarially trained SONet (abstract-level only), concluding that ODE robustness mainly comes from solver gradient obfuscation. — [arXiv 2009.13145](https://arxiv.org/abs/2009.13145)

### Inferences
- For SODE-Guard, the realistic upside of bolting a SODEF-style equilibrium module onto an adversarially trained encoder is a few points of worst-case macro-F1 at most. Its robustness must be verified with surrogate-loss transfer and gradient-free attacks. Adversarial training of the encoder is the component that actually carries robustness.
- SODEF freezes the extractor and trains only the ODE block and head. This keeps the adversarially trained features intact and is cheap to try on the existing GraphSAGE/MLP encoder.

### Gaps
- I found no AutoAttack numbers for TisODE+AT, LyaNet+AT or Neural SDE+AT. I found no RobustBench entry for any ODE/SDE model, but did not query RobustBench directly.

## Q4. Noise as augmentation vs. as a defence; learned sigma collapsing to zero, and remedies

### Takeaway
The literature treats learned noise scales as prone to shrinking to zero unless training pressure (adversarial training in PNI) or an explicit noise-promoting regulariser (Learn2Perturb-style) holds them up. Even when noise survives training, EOT evaluation removes most of its apparent test-time benefit. Noise is better used as a training-time regulariser or certified-smoothing mechanism than as a stochastic defence.

### Cited Findings
- PNI learns the noise scale jointly with adversarial training (min-max), which provides the pressure that keeps the noise non-zero. — [CVPR 2019 PNI](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Parametric_Noise_Injection_Trainable_Randomness_to_Improve_Deep_Neural_Network_CVPR_2019_paper.pdf)
- Follow-up approaches "include some noise-promoting regulariser to prevent the noise from shrinking to zero during training". Learn2Perturb learns parameter-wise noise strength per layer. — [arXiv 2206.01705](https://arxiv.org/pdf/2206.01705v1.pdf) (search snippet)
- PNI is still vulnerable to EOT gradient estimation, so a learned and non-collapsed noise level does not by itself give adaptive robustness. — [arXiv 2206.01705](https://arxiv.org/pdf/2206.01705v1.pdf)
- In Neural SDE (Liu et al.), noise is framed as a regulariser (an SDE generalisation of dropout and Gaussian noise) that stabilises otherwise unstable ODE dynamics. — [arXiv 1906.02355](https://arxiv.org/abs/1906.02355v1)

### Inferences
- The SODE-Guard observation of sigma collapsing to zero is the expected outcome when g is trained on clean cross-entropy alone, because noise only hurts the clean training loss. Remedies, from most to least supported by the sources above:
  1. Train g under adversarial training (PNI-style min-max) so noise has a robustness payoff.
  2. Add a noise-promoting regulariser or impose a fixed floor, e.g. g = sigma_min + softplus(.).
  3. Use a fixed, non-learned sigma or schedule.
  4. Move noise to input space and treat it as randomised smoothing with certification. Randomised smoothing was not researched here, so its applicability to tabular flows is an inference.
- Whatever remedy is used, evaluation must keep EOT and expected-margin attacks. The PNI evidence shows that non-zero noise does not imply robustness.

### Gaps
- I did not retrieve primary sources quantifying sigma collapse (e.g., the PNI ablation of noise without adversarial training) or randomised-smoothing results on tabular IDS data.

## Q5. Neural ODE/SDE applied to intrusion detection and their robustness evaluations

### Takeaway
I found no peer-reviewed neural-SDE or neural-ODE intrusion-detection paper with an adversarial-robustness evaluation, let alone an adaptive one. "SDE-TGNN" could not be located. The nearest works are probabilistic temporal-graph IDS models aimed at noise robustness, not adversarial robustness.

### Cited Findings
- "Noise Robust One-Class Intrusion Detection on Dynamic Graphs" (arXiv 2508.14192) proposes a probabilistic Temporal Graph Network SVDD (TGN-SVDD). It models flows as edge events in a dynamic graph to improve detection under input noise. It does not use an SDE. — [arXiv 2508.14192](https://arxiv.org/html/2508.14192v1)
- Graph SPDE uncertainty estimation ("Uncertainty Estimation on Graphs with Structure Informed Stochastic Partial Differential Equations") is a general graph-learning method, not IDS. — [SlidesLive](https://slideslive.com/39046720/uncertainty-estimation-on-graphs-with-structure-informed-spde?locale=en); [aitopics](https://aitopics.org/doc/arxivorg:8A56D127)
- Other GNN-based NIDS works (hybrid GCN+Transformer; GraphIDS masked autoencoder) do not appear to use ODE/SDE dynamics. — [PMC12822977](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12822977/); [arXiv 2509.16625](https://arxiv.org/pdf/2509.16625v5)

### Inferences
- An adaptive robustness study of ODE/SDE flow classifiers on tabular network data appears to be an open niche. A rigorous negative result (SODE-Guard approximately equal to ODE approximately equal to MLP under EOT and expected-margin attacks) would itself be a contribution, consistent with Huang et al. and Croce et al. on images.

### Gaps
- "SDE-TGNN" and temporal-graph SDE IDS models: no source found. They may not exist under that name, or may be in venues not indexed by the search.
- No neural ODE/SDE IDS work with AutoAttack, EOT or transfer evaluations was found.
