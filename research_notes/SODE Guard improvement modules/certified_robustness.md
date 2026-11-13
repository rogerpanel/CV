# Certified-Robustness Training and Certificate-Time Modules for a Continuous-Depth (Neural ODE/SDE) Classifier on Tabular Network-Flow Data (SODE-Guard)

Research-method note: web *fetching* was blocked in this session (hosts ndss-symposium.org and api.crossref.org failed DNS resolution under the session's network allowlist). Findings below come from search-engine extracts of primary sources, plus arXiv links to canonical papers. Numbers are marked **[verified via search extract]** when a search result quoted them. Numbers marked **[recalled, unverified]** must be checked against the paper before they are quoted in a report. Where neither applies, the claim is under Gaps.

Baseline from the brief (ICS3D, 70k flows, 9 classes, 75 z-scored features, l2 radii 0/0.05/0.1/0.273):
SODE-Guard 0.610/0.538/0.294/0.096 (L=11.5); NODE 0.618/0.608/0.590/0.280 (L=5.1); Lipschitz MLP 0.618/0.608/0.590/0.422 (L=1.4); RS sigma=0.25 0.636/0.628/0.628/0.606.

---

## Q1. Randomized-smoothing (RS) training modules, adaptive/anisotropic smoothing for traffic, and whether the SDE's own Brownian noise can serve as the smoothing noise

### Takeaway
RS training add-ons (SmoothAdv, MACER, Consistency, SmoothMix, CAT-RS) each add only about 0.01-0.03 ACR over each other at sigma=0.25 on CIFAR-10. The large jump comes from switching the certificate from a global-Lipschitz bound to the RS (Neyman-Pearson) certificate at all. I found no published work that certifies a neural SDE by treating its Brownian noise as smoothing noise. A construction that makes it exactly valid does follow from RS theory: a pure-diffusion prefix on the input, or latent noise plus a Lipschitz bound on the pre-noise encoder (PixelDP-style). This is the highest-leverage module for SODE-Guard.

### Cited Findings
**Core RS and training modules**
- Cohen et al. (ICML 2019): Gaussian-noise smoothing gives a tight l2 certificate, R = sigma/2 (Phi^-1(pA) - Phi^-1(pB)). The base model is trained with Gaussian augmentation at the same sigma. The certificate is a Monte-Carlo one: a Clopper-Pearson lower bound on pA at confidence 1-alpha, with abstention when it fails — [Cohen et al. 2019](https://arxiv.org/pdf/1902.02918)
- CAT-RS paper (AAAI 2023), CIFAR-10 sigma=0.25 table [verified via search extract]. ACR: MACER 0.531, Consistency 0.552, SmoothMix 0.553, CAT-RS 0.562. Clean (r=0) accuracy: MACER 79.5, SmoothMix 77.1, CAT-RS 76.3, Consistency 75.8. CAT-RS row: ACR 0.562, certified accuracy 76.3/68.1/58.8/48.2 at r=0/0.25/0.5/0.75 — [CAT-RS (Jeong et al.) arXiv](https://arxiv.org/pdf/2212.09000); [AAAI version](https://ojs.aaai.org/index.php/AAAI/article/download/25968/25740)
- The same methods rank differently under other protocols. The AdvMacer paper reports CIFAR-10 sigma=0.25 ACR of MACER 0.518, Consistency 0.535, SmoothMix 0.545 on 500 test images. SPLITZ reports certified accuracy at r=0.25 of MACER 71.0% vs SmoothMix 67.9%. Conclusion: differences between RS training methods sit within protocol noise — [AdvMacer / "Promoting Robustness of RS"](https://arxiv.org/pdf/2310.07780); [SPLITZ](https://arxiv.org/html/2407.02811)
- SmoothMix (NeurIPS 2021) trains with mixup between clean inputs and adversarial examples on the smoothed classifier, aiming at a better accuracy/robustness trade-off — [SmoothMix](https://arxiv.org/pdf/2111.09277)
- SmoothAdv (Salman et al., NeurIPS 2019) trains on adversarial examples of the smoothed classifier — [Salman et al. 2019](https://arxiv.org/abs/1906.04584). MACER (ICLR 2020) adds a differentiable certified-radius-maximization loss, with no adversarial attack in training — [MACER](https://arxiv.org/abs/2001.02378). Consistency regularization (Jeong & Shin, NeurIPS 2020) adds a KL term that makes predictions under different noise draws agree — [Jeong & Shin 2020](https://arxiv.org/abs/2006.04062). Denoised smoothing (Salman et al., NeurIPS 2020) puts a denoiser in front of an off-the-shelf classifier — [Salman et al. 2020](https://arxiv.org/abs/2003.01908). Method descriptions are from the primary abstracts; I did not verify their numbers in this session.

**Input-dependent / anisotropic smoothing**
- ANCER chooses an axis-aligned anisotropic (per-dimension) Gaussian for each input to maximize certified volume. It reports SOTA l1/l2 certified accuracy on CIFAR-10/ImageNet in the data-dependent setting. Soundness caveat: the memory-based variant makes the classifier depend on input order, which leaves it ambiguous what is being certified — [ANCER](https://arxiv.org/pdf/2107.04570); [ANCER html](https://arxiv.org/html/2107.04570); data-dependent RS: [Alfarra et al.](https://arxiv.org/pdf/2012.04351); non-axis-aligned extension: [RANCER WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Rumezhak_RANCER_Non-Axis_Aligned_Anisotropic_Certification_With_Randomized_Smoothing_WACV_2023_paper.pdf)
- BARS (NDSS 2023, Wang et al., Tsinghua) is RS for DL traffic-analysis systems. It uses "optimized smoothing noise converging on the classification boundary", produced by a "Distribution Transformer" that generates per-feature noise. It is designed for heterogeneous features and claims tighter guarantees than baseline RS. Code: KaiWangGitHub/BARS — [BARS NDSS page](https://ndss-symposium.org/ndss-paper/bars-local-robustness-certification-for-deep-learning-based-traffic-analysis-systems); [paper PDF](https://www.ndss-symposium.org/wp-content/uploads/2023-508-paper.pdf)
- MARS ("Multi-order Adaptive Randomized Smoothing"; TrustCom 2024, journal version in ACM TOPS, DOI 10.1145/3715121) certifies robustness of DNN-based network intrusion detectors. Abstract: +12.23% l2 certified radius vs the SOTA method (BARS), better detection under evasion attacks, better accuracy under natural corruption, and a dimension-wise analysis of certified radii — [TrustCom slides](https://mengdiehuang.github.io/slides/TrustCom2024-MARS-15min-Presentation-Slides-20241218.pdf); [Crossref record](https://api.crossref.org/works/10.1145%2F3715121); [abstract listing](https://www.bohrium.com/scholar/95397t04/Xiaofeng_Chen)
- Traffic-Aware RS (2026 arXiv preprint, not peer-reviewed) adds Gaussian noise only to the features an attacker can modify, in both fine-tuning and certification. It reports 55-100% certified accuracy at an L-inf-equivalent eps=0.05 on CIC-IDS-2018 and HIKARI-2021. Clean-trained models certify poorly, and noise-augmented fine-tuning is essential. Empirical robustness exceeds certified accuracy by 16-32 points, attributed to a conservative Clopper-Pearson bound — [Traffic-Aware RS](https://arxiv.org/abs/2607.13801v1); [html](https://arxiv.org/html/2607.13801)

**Latent-space noise / SDE noise as smoothing**
- Latent-space smoothing (Zeng, Su, Huang): split the net into encoder + classifier and add Gaussian noise to a compact latent code instead of the input, to cut the cost of high-dimensional smoothing — [arXiv 2108.00491](https://arxiv.org/abs/2108.00491v1)
- PixelDP (Lecuyer et al., S&P 2019) puts the noise layer *inside* the network. The input-space guarantee comes from bounding the sensitivity (Lipschitz constant) of the pre-noise layers — [PixelDP](https://arxiv.org/abs/1802.03471)
- Neural SDE (Liu et al., arXiv 1906.02355; CVPR 2020 "How Does Noise Help Robustness?") adds a diffusion term to a NODE that unifies dropout, additive, and multiplicative noise injection. The authors argue the stochastic term limits growth of input perturbations, report better results on most common-corruption categories, and give a theoretical (non-certified) analysis. Nothing ties it to RS certificates — [Neural SDE arXiv](https://www.alphaxiv.org/abs/1906.02355v1); [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_How_Does_Noise_Help_Robustness_Explanation_and_Exploration_under_the_CVPR_2020_paper.pdf)
- Searches for "certified robustness neural SDE randomized smoothing Brownian noise" found no paper that certifies neural SDEs by smoothing the Brownian noise. Only input-space RS, latent RS, and DEQ/diffusion-related RS turned up — [search context: Cohen](https://arxiv.org/pdf/1902.02918); [latent RS](https://arxiv.org/abs/2108.00491v1)
- Diffusion-model denoised smoothing ("(Certified!!) Adversarial Robustness for Free!", Carlini et al., ICLR 2023) shows a pretrained diffusion denoiser + off-the-shelf classifier gives strong RS certificates — [Carlini et al.](https://arxiv.org/abs/2206.10550) (method from abstract; numbers not verified here)

### Inferences
- **Biggest single lever: certify SODE-Guard with RS instead of Theorem A.** The brief's own numbers show RS at sigma=0.25 holding 0.606 at r=0.273, vs 0.096 for the Lipschitz certificate. The RS certificate needs no network Lipschitz bound. Salman et al. 2019 show Phi^-1 of the smoothed class probability is (1/sigma)-Lipschitz for *any* base classifier, so the exp(K T) factor disappears.
- **Making the SDE noise itself the smoothing noise (sound construction).** Cohen's Neyman-Pearson argument holds for a *randomized* base classifier when the extra randomness is independent of x. So:
  - (a) Add a pure-diffusion prefix: on t in [0, tau], set drift f=0 and diffusion sigma_0*I on the input (or on an isometric embedding of it). Then z(tau) = x + sigma_0 sqrt(tau) eps exactly, which is Gaussian RS with sigma_RS = sigma_0 sqrt(tau). All later drift and Brownian noise become part of the randomized base classifier. Certify with Clopper-Pearson on the top class and R = sigma_RS * Phi^-1(pA_lower), or the two-class formula.
  - (b) If noise enters after an encoder h with Lipschitz L_h (PixelDP-style), the input radius is R_latent / L_h. Keep h 1-Lipschitz, e.g. orthogonal or SLL (see Q2).
  - State-dependent diffusion g(z) at the first noise injection breaks the Gaussian-shift structure. The noise used for the certificate must be additive and isotropic, or a fixed known anisotropic covariance for ANCER-style ellipsoidal certificates, but then it cannot depend on x unless one accepts the ANCER caveats.
- **Train for it.** Gaussian augmentation at sigma_RS is the minimum. Add Consistency or CAT-RS on the SDE's MC samples; they are cheap because the SDE already draws several samples. SmoothAdv/SmoothMix add attack cost. Expect training-method choice to matter less than sigma choice and the switch to RS.
- **Traffic-specific.** Feature-wise (anisotropic) noise à la BARS/MARS/Traffic-aware RS fits z-scored heterogeneous flow features. Restrict or scale noise to attacker-modifiable features, but certify the threat model honestly: an ellipsoidal certificate is not an l2-ball certificate.
- **Soundness of the current SODE-Guard MC certificate.** Theorem A estimates *mean logits* with Hoeffding/Bernstein. This is sound only if (i) every sample path's logit map is L-Lipschitz with the same constant (true for additive noise under a fixed Brownian path), (ii) the margin uses lower/upper confidence bounds with a union bound over the 9 classes, and (iii) logits are bounded for Hoeffding, or variance is controlled for Bernstein. A reviewer will check all three.

### Gaps
- Exact BARS/MARS certified-accuracy or radius tables (datasets, sigma, alpha, n) could not be fetched: the NDSS PDF host was blocked. From memory, BARS evaluated traffic systems such as Kitsune, Whisper, and DF website fingerprinting; this is unverified.
- No peer-reviewed source found on certifying neural SDEs via their Brownian noise. The construction above is my inference from RS theory, not a published result.
- Exact SmoothAdv/MACER/Consistency/denoised-smoothing numbers on tabular data were not found. All verified numbers are CIFAR-10.

---

## Q2. Lipschitz-by-design layers and margin losses

### Takeaway
Modern 1-Lipschitz architectures (SLL, LiResNet, AOL, SOC/Cayley, GroupSort/MaxMin) combined with Lipschitz-margin or GloRo-style losses are the deterministic alternative. For SODE-Guard they matter mainly as the readout W_psi and encoder h, and for constraining the ODE drift (Q3). Your Lipschitz MLP (L=1.4) already beats the SDE at r=0.273 for exactly this reason.

### Cited Findings
- SLL (Araujo et al., ICLR 2023) unifies Lipschitz layers through an SDP/algebraic view. The originally reported CIFAR-10 numbers (X-Large: 65.8% certified at eps=36/255) were affected by an implementation bug. The erratum (arXiv Appendix D) reports X-Large at 73.3 / 70.3 / 65.4 / 60.2 (clean / increasing eps) [verified via search extract; column-eps mapping should be checked] — [SLL arXiv](https://arxiv.org/pdf/2303.03169); [html](https://arxiv.org/html/2303.03169)
- LiResNet line / "A Recipe for Improved Certifiable Robustness" (ICLR 2024) surpasses prior SOTA by up to 8.5 pp on CIFAR-10/100 deterministic certified accuracy and lists an SLL baseline at 85.5/75.6/65.6/52.3 on CIFAR-10. Baseline numbers differ between papers, so compare carefully — [Recipe, ICLR 2024](https://arxiv.org/pdf/2310.02513)
- Sandwich layers ("Direct Parameterization of Lipschitz-Bounded Deep Networks") report SLL results on CIFAR-100 and Tiny-ImageNet — [Wang & Manchester 2023](https://arxiv.org/pdf/2301.11526); related structural analysis: [Exploiting Connections between Lipschitz Structures, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/4462db5eee6823b2abad0d1f955e187a-Paper-Conference.pdf)
- Local Lipschitz bounds during training (Huang et al., NeurIPS 2021) give a tighter-than-global bound and improved certified accuracy on CIFAR-10 — [Huang et al. 2021](https://proceedings.neurips.cc/paper/2021/file/c055dcc749c2632fd4dd806301f05ba6-Paper.pdf)
- Dynamic margin maximization + improved Lipschitz regularization (2023) trains with margin-based objectives and reports certified radii on CIFAR-10 — [arXiv 2310.00116](https://arxiv.org/html/2310.00116)
- LipNeXt (2026) scales Lipschitz-based certified robustness further — [LipNeXt](https://arxiv.org/pdf/2601.18513); N-activations increase 1-Lipschitz expressiveness — [arXiv 2311.06103](https://arxiv.org/pdf/2311.06103)
- Canonical layer and loss papers (methods only; numbers not verified this session): Lipschitz-Margin Training, which adds sqrt(2)·L·eps to non-target logits ([Tsuzuku et al., NeurIPS 2018](https://arxiv.org/abs/1802.04034)); GloRo nets, which add a "bottom" logit computed from the global Lipschitz bound ([Leino et al., ICML 2021](https://arxiv.org/abs/2102.08452)); GroupSort/MaxMin activations ([Anil et al., ICML 2019](https://arxiv.org/abs/1811.05381)); Cayley orthogonal convs ([Trockman & Kolter, ICLR 2021](https://arxiv.org/abs/2104.07167)); SOC ([Singla & Feizi, ICML 2021](https://arxiv.org/abs/2105.11417)); BCOP ([Li et al., NeurIPS 2019](https://arxiv.org/abs/1911.00937)); AOL ([Prach & Lampert, ECCV 2022](https://arxiv.org/abs/2208.03160)).

### Inferences
- For SODE-Guard, the cheap fixes are:
  - (i) Parameterize W_psi and h as orthogonal/AOL/SLL, so that ||W_psi|| L_h ≤ 1 exactly rather than estimated.
  - (ii) Train with the LMT/GloRo margin loss using the *same* L as Theorem A, so training optimizes the certified margin margin/(sqrt(2) L) directly. GloRo-style training is the deterministic analogue of MACER.
  - (iii) Use MaxMin/GroupSort in the drift network, which keeps gradient norm while staying 1-Lipschitz.
- For tabular 75-dim inputs, a 1-Lipschitz MLP with margin training is a strong, simple baseline. The brief's L=1.4 MLP shows this. The SDE has to bring its L to roughly 1-2 to compete under a Lipschitz certificate.

### Gaps
- Verified GloRo/LMT certified-accuracy numbers (e.g., CIFAR-10 at 36/255) were not retrieved.
- No source reporting 1-Lipschitz layers on tabular NIDS data was found.

---

## Q3. Contractive / stable neural ODEs and contraction-based certificates

### Takeaway
Contraction is the principled fix for exp((K_f + ...)T). With an l2 (or weighted/non-Euclidean) log-norm bound mu(∂f/∂z) ≤ c, the flow's Lipschitz constant is exp(cT). With c ≤ 0 this is ≤ 1, so the ODE block stops amplifying. Several works enforce this (contractive Hamiltonian NODEs, NEMON, log-norm weight projection), and FI-ODE gives a *certified* NODE result (42.27% on CIFAR-10 at l2 eps=0.141). Lyapunov-stable defenses (SODEF) report only empirical robustness, and their numbers have been challenged.

### Cited Findings
- Improving NODE robustness with minimal weight perturbation (2025): characterizes the one-sided Lipschitz constant via the log-norm of the vector-field Jacobian, and projects weights to the closest matrix with mu_2(D B) = delta, which bounds how input perturbations amplify — [arXiv 2501.10740](https://arxiv.org/pdf/2501.10740)
- Contractive Hamiltonian NODEs (Zakwan, Xu, Ferrari-Trecate) use contraction theory to make NODE classifiers robust to small input perturbations (e.g., white noise) — [arXiv 2203.11805](https://arxiv.org/pdf/2203.11805v3.pdf); [EPFL summary](https://actu.epfl.ch/news/contractive-neural-networks-are-robust-against-noi); convolutional follow-up with contractivity-promoting regularization: [arXiv 2508.11432](https://www.arxiv.org/pdf/2508.11432)
- FI-ODE (Huang et al.): certified robustness of NODEs via forward invariance + Lyapunov training + Lipschitz restriction + adaptive sampling. CIFAR-10 l2 eps=0.141: 42.27% certified vs 41.43% for a LyaNet baseline with Lipschitz restriction [verified via search extract] — [FI-ODE](https://arxiv.org/pdf/2210.16940); [abs](https://arxiv.org/abs/2210.16940v2)
- LyaNet (Rodriguez et al., ICML 2022) trains NODE classifiers with a Lyapunov loss for fast, stable convergence to the correct class — [LyaNet](https://arxiv.org/abs/2202.02526) (method only)
- SODEF (Kang et al., NeurIPS 2021) regularizes features to sit near Lyapunov-stable equilibria. Reported AutoAttack robust accuracy on CIFAR-10 with TRADES: 57.76% (l-inf) / 67.75% (l2) vs TRADES 53.69% / 59.42%; with Rebuffi2021 it goes from 66.58% to over 70%. **Caveats:** this is empirical, not certified. Croce et al. show AutoAttack returns originals when no adversarial is found, and transfer attacks drive SODEF below plain TRADES. A later paper suggests stabilized-NODE robustness may come from obfuscated gradients — [SODEF](https://arxiv.org/abs/2110.12976); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/7d5430cf85f78c4b7aa09813b14bce0d-Paper.pdf); [code](https://github.com/kangqiyu/SODEF); [Croce et al., "Evaluating the Adversarial Robustness of Adaptive Test-time Defenses"](https://arxiv.org/pdf/2202.13711); [Aligned stability in NODEs, 2025](https://arxiv.org/pdf/2509.21879)
- Non-Euclidean contraction:
  - NEMON (Jafarpour, Davydov, Proskurnikov, Bullo, NeurIPS 2021) builds well-posed, robust implicit networks from l-inf contraction, with l-inf input-output Lipschitz bounds — [arXiv 2106.03194](https://arxiv.org/pdf/2106.03194)
  - Robust training + verification of implicit nets via non-Euclidean contraction, MNIST experiments — [arXiv 2208.03889](https://arxiv.org/pdf/2208.03889); [ICML-workshop PDF](https://coogan.ece.gatech.edu/papers/pdf/jafarpour2022icml.pdf)
  - Contraction analysis of continuous-time (firing-rate/Hopfield) NNs — [Davydov et al. arXiv 2110.08298](https://arxiv.org/pdf/2110.08298)
  - Nonlinear separation principle via contraction, with NN applications — [arXiv 2604.15238](https://arxiv.org/pdf/2604.15238)
- Monotone operator equilibrium networks (monDEQ) guarantee unique fixed points via a strong-monotonicity parameterization; their Lipschitz variants give certificates — [Winston & Kolter, NeurIPS 2020](https://arxiv.org/abs/2006.08591) (method only)

### Inferences
- **Replace K_f by the log-norm.** For dz = f(z) dt + sigma dW with *additive* (state-independent) noise, two solutions driven by the same Brownian path satisfy d(z1-z2)/dt = f(z1) - f(z2). So |z1(T) - z2(T)| ≤ exp(mu T) |z1(0) - z2(0)| pathwise, where mu = sup mu_2(∂f/∂z) ≤ K_f. The 0.5 K_g^2 term in Theorem A only appears with multiplicative (state-dependent) diffusion. Switching SODE-Guard to additive diffusion removes it. Enforcing mu ≤ 0, e.g. drift f(z) = -alpha z + W2 sigma(W1 z) with alpha ≥ ||W2|| ||W1|| L_sigma, or a skew-symmetric/Hamiltonian parameterization with damping, makes the flow non-expansive. Then L ≤ ||W_psi|| L_h, which can be ≤ 1. This is my derivation (standard contraction theory) and should be checked.
- Contraction trades expressivity for certificate tightness. A strictly contracting flow with fixed T forgets input differences, which is why FI-ODE/LyaNet rely on class-dependent attractors plus a margin readout. Expect clean accuracy to drop slightly. Tune c ≈ 0 (non-expansive) rather than strongly negative.
- Weighted-l2 or l-inf log-norms (Bullo school) can be much tighter than the l2 spectral bound for sparse or heterogeneous tabular drifts. l-inf contraction gives l-inf certificates, which suits feature-wise flow perturbations.

### Gaps
- No published tight comparison between exp(K_f T) bounds and true empirical Lipschitz constants of trained NODEs was found.
- No contraction-certified neural *SDE* classifier was found. Stochastic contraction (e.g., Pham & Slotine) was not retrieved in this session.
- FI-ODE numbers are image-domain only. No tabular contraction-certified results were found.

---

## Q4. Bound propagation (IBP / CROWN-IBP / alpha-beta-CROWN) and NODE verification tools

### Takeaway
For a 75-dim tabular input and small MLP drifts, bound propagation is feasible. GAINS (ICLR 2023) is the only established certified-*training* method for NODEs, reaching up to 89% certified accuracy on MNIST/F-MNIST. Reachability verifiers (NNV 2.0, GoTube, TNODEV, CORA) can certify trained NODEs but are sound-and-incomplete (UNKNOWN verdicts) and costly. GoTube's guarantees are probabilistic, not deterministic.

### Cited Findings
- GAINS (Zeqiri, Müller, Fischer, Vechev, ICLR 2023) combines controlled adaptive ODE solvers with discrete variable steps, a trajectory-graph representation, and an abstraction algorithm (CURLS) on it. Verification cost drops from exponential to O(d + T^2 log^2 T). Certified training reaches up to 89% certified accuracy (e.g., MNIST eps=0.1) at a moderate standard-accuracy cost (e.g., 98.8% to 95.5% MNIST). Training stays expensive (multiple trajectories) — [GAINS arXiv](https://arxiv.org/pdf/2303.05246); [OpenReview](https://openreview.net/forum?id=KyoVpYvWWnK); [code](https://github.com/eth-sri/GAINS)
- NNV 2.0 (CAV 2023): star-set reachability with neural-ODE support, evaluated on MNIST GNODE robustness (CNODE_S/CNODE_M) — [NNV 2.0](https://link.springer.com/chapter/10.1007/978-3-031-37703-7_19); [Reachability of general NODEs, arXiv 2207.06531](https://arxiv.org/pdf/2207.06531)
- GoTube: stochastic Lagrangian reachability with Lipschitz balls, giving *probabilistic* (statistical) reach-tube guarantees — [GoTube](https://arxiv.org/abs/2107.08467) (method from search context)
- TNODEV (2026 preprint) is a sound NODE verifier: falsification, interval reachability via continuous-time mixed monotonicity, and iterative input-splitting refinement. It is benchmarked against NNV 2.0 and CORA on MNIST NODE classifiers — [TNODEV](https://arxiv.org/abs/2606.16567)
- An applied NODE classifier verified with NNV 2.0 notes that UNKNOWN is the standard inconclusive outcome of sound-but-incomplete reachability verifiers — [plankton classifier, 2026](https://arxiv.org/html/2607.04453v1)
- VNN-COMP overview of verifiers including alpha-beta-CROWN — [VNN-COMP 2021](https://arxiv.org/pdf/2109.00498). IBP training: [Gowal et al. 2018](https://arxiv.org/abs/1810.12715); CROWN-IBP: [Zhang et al., ICLR 2020](https://arxiv.org/abs/1906.06316); beta-CROWN: [Wang et al., NeurIPS 2021](https://arxiv.org/abs/2103.06624) (methods only; numbers not verified)

### Inferences
- A practical SODE-Guard path: discretize with a fixed-step solver (Euler/RK4, a few steps), treat it as a residual MLP, and train with CROWN-IBP or IBP plus a final-layer margin. Then certify the *deterministic* NODE (mean drift, or the drift at a fixed Brownian sample) with alpha-beta-CROWN. A local (input-specific) bound from CROWN is often far tighter than a global Lipschitz product.
- Soundness: certify the *discretized* model actually deployed. Bounds on the continuous ODE do not transfer to a different solver or step size unless solver error is bounded (GAINS addresses this by certifying its own solver). For the SDE, a CROWN certificate holds per Brownian sample; aggregate via RS (Q1) or report "certified for fixed noise seed".

### Gaps
- No IBP/CROWN certified-accuracy results on CIC-IDS/UNSW/ICS-type tabular NIDS data were found in this session.
- No verifier for neural SDEs was found.

---

## Q5. Typical reported certified accuracy at small radii on tabular / NIDS data

### Takeaway
Published NIDS certification is almost entirely RS-based: BARS, MARS, and Traffic-aware RS. Reported numbers vary widely (55-100% certified accuracy at small L-inf-equivalent budgets in one 2026 preprint). Standardized tabular l2 benchmarks are lacking, so the brief's internal RS baseline (0.606 at r=0.273) is the most relevant comparator.

### Cited Findings
- Traffic-aware RS (preprint): 55-100% certified accuracy at L-inf-equivalent eps=0.05 on CIC-IDS-2018 and HIKARI-2021. Clean-trained models certify poorly. Clopper-Pearson makes certificates conservative by 16-32 points vs empirical robustness — [arXiv 2607.13801](https://arxiv.org/abs/2607.13801v1)
- MARS: +12.23% l2 certified radius over the prior SOTA (BARS) on NIDS — [MARS slides](https://mengdiehuang.github.io/slides/TrustCom2024-MARS-15min-Presentation-Slides-20241218.pdf); [Crossref](https://api.crossref.org/works/10.1145%2F3715121)
- BARS: tighter certificates than baseline RS for traffic-analysis systems; exact numbers not retrieved — [BARS](https://ndss-symposium.org/ndss-paper/bars-local-robustness-certification-for-deep-learning-based-traffic-analysis-systems)
- A conformal-IDS preprint with traffic-aware restrictions gives only statistical (coverage) guarantees, not robustness certificates — [arXiv 2609.19241](https://arxiv.org/pdf/2609.19241)

### Inferences
- With 75 z-scored features, l2 r=0.273 is about 0.03 std per feature if spread evenly. That is small, which explains why RS at sigma=0.25 barely loses accuracy. To be competitive, SODE-Guard should report RS-style certificates (sigma in {0.12, 0.25, 0.5}) with n0=100, n=10^4-10^5, alpha=0.001 (Cohen's protocol), plus ACR.
- Priority order for SODE-Guard:
  1. RS certificate using an additive-noise prefix as smoothing noise, with Gaussian augmentation + Consistency/CAT-RS training (Q1).
  2. Additive diffusion + log-norm (contraction) constraint on the drift + orthogonal/SLL readout + GloRo/LMT margin loss, to cut L from 11.5 toward 1 for the deterministic certificate (Q2-Q3).
  3. CROWN-IBP training/verification on the discretized drift for local bounds (Q4).
  4. Anisotropic feature-wise noise (BARS/MARS-style) only with an explicitly stated ellipsoidal threat model.

### Gaps
- No peer-reviewed tabular-NIDS l2 certified-accuracy tables at radii comparable to 0.05-0.273 were retrieved (BARS/MARS full texts were unreachable).
- Hosts needed to verify: ndss-symposium.org (BARS PDF), api.crossref.org / dl.acm.org (MARS TOPS). These must be allowed in the session network settings to fetch.
