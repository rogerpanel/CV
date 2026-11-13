# Imbalance-handling and tabular-DL modules for raising macro-F1 of SODE-Guard on ICS3D (plus strong NIDS baselines)

Method note: in this session, full-text fetching from arxiv.org and kaggle.com was blocked by the sandbox network policy (DNS failure on both hosts). All findings below come from web-search result snippets and summaries of primary sources. Exact table numbers that could not be read from the paper are listed under Gaps, not guessed. Numbers quoted from NIDS papers come from snippets and should be spot-checked against the PDFs before they go into a paper.

## Q1. Loss and sampling modules for imbalance (focal, CB, LDAM-DRW, logit adjustment, balanced softmax, DRW, class-balanced sampling, SMOTE/ADASYN/CTGAN, post-hoc tuning): which give the largest macro-F1 gains on NIDS?

### Takeaway
No controlled NIDS study compares all of these modules head-to-head. The NIDS numbers that exist (mostly from heterogeneous setups) show gains of about +0.07 to +0.14 macro-F1 from loss re-weighting or margins, and about +0.08 to +0.10 from CTGAN over no oversampling on NetFlow data. SMOTE/ADASYN add only about 1-2 points on top of an RF that is already strong. Long-tail theory (Cao 2019; Menon 2021) predicts that two-stage schemes (plain ERM first, then re-weighting or margins via DRW, or a post-hoc logit/prior correction tuned on validation) beat re-weighting from epoch 0. For SODE-Guard's 10-epoch CE setting, that makes logit adjustment and DRW the cheapest first modules to try.

### Cited Findings
**Long-tail method definitions (primary sources)**
- Class-Balanced (CB) loss (Cui et al., CVPR 2019) weights each class by the inverse "effective number" (1 − β^n)/(1 − β). β is searched over {0.9, 0.99, 0.999, 0.9999} and combined with softmax, sigmoid or focal loss. On long-tailed CIFAR-10 the best β was 0.9999 "unanimously", which is close to inverse-frequency weighting. On CIFAR-100 only small β helped and large β hurt. — [Cui et al. 2019, arXiv 1901.05555](https://arxiv.org/pdf/1901.05555); [CVF open access](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)
- LDAM-DRW (Cao et al., NeurIPS 2019) uses a label-distribution-aware margin loss (larger margins for rare classes) that replaces CE and can be combined with re-weighting or re-sampling. DRW trains with unweighted LDAM until the LR anneals, then switches to re-weighted LDAM at a small LR. The authors found deferred re-weighting or re-sampling more effective than re-weighting or re-sampling from the start. — [Cao et al. 2019, NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html)
- Reported reference numbers: LDAM-DRW reaches 77.03% top-1 on CIFAR-10-LT at imbalance ratio 100, 73.52% at ratio 200 and 81.03% at ratio 50 (as tabulated by a later comparison paper). — [Gaussian Clouded Logit Adjustment, arXiv 2305.11733](https://arxiv.org/pdf/2305.11733)
- Logit adjustment (Menon et al., ICLR 2021) has two forms: post-hoc (subtract τ·log prior from the logits at inference) and a logit-adjusted loss. On CIFAR-10 the paper reports an 8% relative reduction in balanced error for post-hoc logit adjustment versus weight normalisation. — [Menon et al., arXiv 2007.07314](https://arxiv.org/pdf/2007.07314); [ICLR 2021 slides](https://iclr.cc/media/Slides/iclr/2021/virtual(05-16-00)-05-16-00UTC-2675-long-tail_learn.pdf)
- A later method that uses Balanced-Softmax and logit adjustment as base margins adds +0.19% and +3.05% on CIFAR-10-LT over those two bases. This indicates the two are close to each other and are both standard strong baselines. — [arXiv 2305.11733](https://arxiv.org/pdf/2305.11733)

**NIDS-specific results**
- CIC-IDS2017 with a knowledge-guided loss: a baseline CNN completely missed rare classes such as Heartbleed (F1 = 0.0). The proposed loss raised macro-F1 from 0.6180 to 0.7560 (+0.138). — [search summary of IJEECS margin/imbalance-loss literature](https://ijeecs.iaescore.com/index.php/IJEECS/article/view/45292) (exact paper attribution unverified; see Gaps)
- NF-UNSW-NB15-v2: class-balanced weighting combined with LDAM-Focal loss reached 80.19% macro-F1. — [search result referencing a typeset.io-hosted imbalance-loss paper](https://typeset.io/pdf/dynamically-weighted-balanced-loss-class-imbalanced-learning-ngwo6u598b.pdf) (attribution approximate)
- Federated CPS study on CIC-IDS2017: focal loss alone gave F1 0.48 and oversampling alone gave 0.62; combining them gave the best result. — [search summary, Tech Science JCS](https://www.techscience.com/JCS/v8n1/66866/html) (attribution approximate)
- CBF-IDS (CNN-BiLSTM with focal loss, Applied Sciences 2023) reports 99.53% accuracy on CIC-IDS2017 and also evaluates NSL-KDD and UNSW-NB15. Macro-F1 was not visible in the snippet. — [MDPI Applied Sciences 13(21):11629](https://mdpi.com/2076-3417/13/21/11629)
- Margin-reciprocal loss is proposed in IJEECS specifically for imbalanced network-anomaly traffic. — [IJEECS 45292](https://ijeecs.iaescore.com/index.php/IJEECS/article/view/45292)

**Oversampling (SMOTE / ADASYN / CTGAN)**
- CIC-IDS2017 with Random Forest: plain RF reached 93.4% macro-F1, SMOTE+RF 94.945% and ADASYN+RF 95.303%, so oversampling adds only about 1.5-1.9 points to an already strong RF. — [ADASYN-RF, arXiv 2105.04301](https://arxiv.org/pdf/2105.04301)
- NF-CSE-CIC-IDS2018-v2: macro-F1 was 0.678 on the original data, 0.697 with SMOTE and 0.780 with CTGAN; the same table lists another oversampling baseline at 0.691 and leaves ADASYN blank. — [CTGAN-enhanced IDS for rare attacks in IoT, arXiv 2502.06031](https://arxiv.org/pdf/2502.06031)
- CSE-CIC-IDS2018: SMOTE, ADASYN and ROS gave F1 of about 93-94%, while GAN-based oversamplers gave about 99%. — [Feature-Regularized GANs, Future Internet 17(5):216](https://www.mdpi.com/1999-5903/17/5/216)
- A CIC-IDS2018 CTGAN study found at least 70% synthetic data was needed to keep performance optimal, and another CTGAN paper warned that excessive augmentation causes overfitting. — [IEEE Xplore 10765501](https://ieeexplore.ieee.org/document/10765501/); [arXiv 2502.06031](https://arxiv.org/pdf/2502.06031)
- Further oversampling+stacking pipelines and hybrid SMOTE-CTGAN pipelines exist. — [arXiv 2401.12262](https://arxiv.org/pdf/2401.12262); [BDCC 10(9):304](https://doi.org/10.3390/bdcc10090304); [S2CGAN-IDS, arXiv 2306.03707](https://arxiv.org/pdf/2306.03707)

### Inferences
- In the long-tail literature, the strongest no-extra-cost modules are logit adjustment (as a loss, or post-hoc with τ tuned on validation macro-F1), Balanced Softmax (mathematically equivalent to the logit-adjusted loss at τ=1), and LDAM-DRW. All three are drop-in changes to SODE-Guard's CE head. Post-hoc logit adjustment needs no retraining, so it can be applied to the existing checkpoints immediately.
- SODE-Guard reaches 90-94% accuracy but only 0.43 macro-F1. That pattern (majority classes right, rare classes near F1 0) matches the "Heartbleed F1=0" failure mode, where the reported loss changes recovered about +0.14 macro-F1. Expect the largest gains to come from (i) making the rare classes visible to the optimizer (re-weighting or class-balanced sampling, ideally deferred as in DRW) and (ii) training longer than 10 epochs. With only 131 samples in the smallest class, 10 epochs means the smallest class contributes very few gradient steps.
- Because CB loss with β→1 converges to inverse-frequency weighting, and the imbalance is about 380:1 (50,000 vs 131), clip the weights or use DRW to avoid instability.
- GAN oversampling showed the biggest reported NIDS gains, but it is costly and risks overfitting. Try SMOTE (cheap, on z-scored or quantile-transformed features) after the loss-level fixes.
- Post-hoc per-class threshold or bias tuning on a validation split, which generalizes logit adjustment, directly optimizes macro-F1. It must use a held-out validation set, not test, to avoid inflated scores.

### Gaps
- I found no single controlled NIDS study that compares focal, CB, LDAM-DRW, logit adjustment and balanced softmax on CIC-IDS2017/2018, UNSW-NB15, CIC-IoT2023 or ToN-IoT under one protocol.
- Exact CIFAR-LT accuracies for logit adjustment and Balanced Softmax could not be read because the arXiv full text was blocked.
- The attribution of the "0.6180 to 0.7560 macro-F1" knowledge-guided-loss result and the "80.19% NF-UNSW-NB15-v2" result to specific papers is approximate, because only aggregated search summaries were available.
- No source was found that quantifies post-hoc threshold tuning gains on NIDS macro-F1.

## Q2. Tabular-DL modules (preprocessing, numerical embeddings, FT-Transformer, TabM, RealMLP, TabPFN v2, TabR) and which simple changes close the gap to GBDTs

### Takeaway
The best-supported findings are these. (1) Trees still win on medium-sized tabular data because NNs are hurt by uninformative features, rotation-invariant processing and irregular target functions (Grinsztajn 2022). (2) Numerical-feature embeddings (piecewise-linear or periodic) let plain MLPs match attention models and close the gap to GBDT on some GBDT-friendly benchmarks (Gorishniy 2022). (3) MLP-based models with efficient ensembling (TabM, ICLR 2025) or tuned defaults and preprocessing (RealMLP, NeurIPS 2024) are the current strongest DL options, and attention or retrieval models do not pay off. TabPFN v2 is strong up to about 10k samples but has documented trouble with large-scale and many-category data. ICS3D (70k rows) exceeds that size.

### Cited Findings
- Grinsztajn, Oyallon and Varoquaux (NeurIPS 2022 Datasets and Benchmarks) built a 45-dataset benchmark. Tree-based models "remain state-of-the-art on medium-sized data (~10K samples) even without accounting for their superior speed." Their three challenges for tabular NNs are robustness to uninformative features, preserving the orientation of the data, and easily learning irregular functions. — [arXiv 2207.08815](https://arxiv.org/abs/2207.08815); [NeurIPS poster](https://nips.cc/virtual/2022/poster/55627)
- Gorishniy, Rubachev and Babenko (NeurIPS 2022) propose two numerical embeddings: piecewise-linear encoding (PLE, based on quantile or target-aware binning) and periodic embeddings (sin/cos with learned frequencies). They beat linear+ReLU embeddings. With them, "simple MLP-like models can perform on par with the attention-based architectures," and DL competes with GBDT on some GBDT-friendly benchmarks (the abstract says "some", not all). — [NeurIPS 2022 proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract.html); [arXiv 2203.05556](https://arxiv.org/pdf/2203.05556v4.pdf)
- TabM (Gorishniy, Kotelnikov and Babenko, ICLR 2025) is a single model that imitates an ensemble of K MLPs (BatchEnsemble-style shared weights with lightweight adapters) and produces K predictions per object. The individual predictions are weak but collectively strong. The authors find parameter-efficient ensembling reduces overfitting and improves optimization dynamics, and conclude that TabM is the best tabular DL model in their large-scale evaluation. They also find that "the complexity of attention- and retrieval-based methods does not pay off" and that MLP-like models are stronger and more practical. — [TabM, ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf); [arXiv 2410.24210](https://arxiv.org/abs/2410.24210)
- A secondary blog reports that on 46 datasets TabM-mini ranks first (average rank 1.7) and TabM second (2.8), ahead of XGBoost, CatBoost and LightGBM. This needs verification against the paper. — [Daily Dose of DS blog](https://blog.dailydoseofds.com/p/tabm-a-powerful-alternative-to-mlp)
- RealMLP and "Better by Default" (Holzmüller, Grinsztajn and Steinwart, NeurIPS 2024) add an improved MLP plus meta-tuned defaults for RealMLP and GBDTs. The defaults were tuned on 118 meta-train datasets and evaluated on 90 disjoint meta-test datasets. RealMLP gives a better time-accuracy tradeoff than other NNs and is competitive with GBDTs. The new GBDT defaults beat library defaults but do not match full HPO. Code is in pytabkit (sklearn interface). — [NeurIPS 2024 abstract](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract.html); [arXiv 2407.04491](https://arxiv.org/abs/2407.04491v1)
- TabPFN v2 (Hollmann et al., Nature 637:319-326, 2025, doi 10.1038/s41586-024-08328-6) is described as outperforming prior methods on datasets up to 10,000 samples with much less training time. — [Zhang et al. reanalysis, arXiv 2505.20003](https://arxiv.org/pdf/2505.20003); [Hugging Face model card](https://huggingface.co/Prior-Labs/TabPFN-v2-clf)
- An independent evaluation on more than 300 datasets found that TabPFN v2 "encounters challenges with large-scale, high-dimensional, and many-category datasets." — [Ye et al., arXiv 2502.17361](https://arxiv.org/pdf/2502.17361)
- TabPFN v2.5 was released on 6 November 2025. — [Wikipedia: TabPFN](https://en.wikipedia.org/wiki/TabPFN)
- On an IIoT SCADA NIDS dataset (2026), macro-F1 was FT-Transformer 0.7070, SAINT 0.6662, XGBoost 0.6276, TabNet 0.5864 and TabTransformer 0.5289. XGBoost was the fastest model (305.6 s). — [Springer Discover AI, s44163-026-02677-z](https://link.springer.com/article/10.1007/s44163-026-02677-z)
- A GitHub reproduction on NSL-KDD and UNSW-NB15 (not peer-reviewed) found XGBoost the strongest and cheapest model. Accuracy was ≥0.95 for every model, so macro-F1 was the discriminating metric. — [GitHub dkansagara12-alt/Network-Intrusion-Detection-System](https://github.com/dkansagara12-alt/Network-Intrusion-Detection-System)

### Inferences
- For heavy-tailed flow features (byte and packet counts, durations, IATs), plain z-scoring leaves extreme outliers. A QuantileTransformer (normal output), a log1p transform before scaling, or RealMLP-style robust scaling with smooth clipping (to be verified in the RealMLP paper) is a cheap, likely-beneficial change. PLE embeddings are quantile-bin based, so they handle heavy tails by construction.
- Priority order for SODE-Guard, from cheapest and best-supported to most expensive. (a) Run many more epochs (10 is very low) with AdamW, a cosine or one-cycle LR schedule, and early stopping on validation macro-F1, not loss. (b) Use quantile or log preprocessing. (c) Put a PLE or periodic embedding front-end before the SDE drift/diffusion nets. (d) Use a TabM-style K-head efficient ensemble. (e) Combine with the imbalance modules from Q1. Items (c) and (d) are architecture-agnostic wrappers, so they fit around the SDE block.
- TabPFN v2 is a worthwhile strong baseline only on a subsample (≤10k rows, stratified to keep the rare classes) or with its ensembling/subsampling extensions. 9 classes is within its supported class count (this limit is from memory and was not verified here).
- Grinsztajn's "uninformative features" bias suggests feature selection (drop constant or near-duplicate CICFlowMeter-style columns among the 75) may help NN models more than GBDTs.

### Gaps
- Exact per-dataset numbers for PLE/periodic embeddings vs GBDT, TabM's ranks, and RealMLP's preprocessing details could not be read because arXiv full text was blocked.
- TabR could not be researched within the tool budget. The TabM paper's statement that retrieval-based methods do not pay off is the only evidence collected here.
- No source was found that isolates the effect of batch size, epochs, AdamW or cosine LR on tabular NN macro-F1. The RealMLP paper is the likely source, but its full text was not read.
- No study applying numerical embeddings or TabM specifically to NIDS data was found.

## Q3. What strong baselines (XGBoost / LightGBM / CatBoost / RF) score on comparable multi-class NIDS datasets, and which data-quality issues inflate scores

### Takeaway
Multi-class macro-F1 for tuned tree ensembles depends heavily on the dataset version and split. Approximate ranges from the sources found: original UNSW-NB15 10-class, about 0.59-0.67; NF-UNSW-NB15-v2, about 0.71; CIC-IDS2017, 0.77 (time-aware split) to 0.93-0.95 (random split); CICIoT2023 34-class, about 0.87; Edge-IIoTset 15-class, about 0.91-0.93. The 0.43 of SODE-Guard is therefore well below any GBDT/RF baseline. Random splits, duplicates and CICIDS2017 labelling errors inflate scores by up to about 0.24 macro-F1.

### Cited Findings
- UNSW-NB15 (original, held-out pooled split), per-class XGBoost results: Backdoor F1 15.92% (P 83.67, R 8.80, n=466) and Worms F1 54.79% (n=35). χ² feature selection (k=80) left these almost unchanged (14.84 / 52.94). Analysis, Backdoor and DoS stay hard. — [IJACSA 16(8) Paper 44](https://thesai.org/Downloads/Volume16No8/Paper_44-Intrusion_Detection_Using_Machine_Learning.pdf)
- UNSW-NB15 macro-F1 under different protocols: XGBoost 67.05% vs RF 65.05% with 5-fold CV and no resampling, and LightGBM+SMOTE 66.37%. — [IIETA reproducible multiclass UNSW-NB15](https://www.iieta.org/download/file/fid/197013)
- UNSW-NB15 held-out test (51,536 records): RF 0.600 and XGBoost 0.596 macro-F1, statistically indistinguishable; Worms and Shellcode estimates are volatile. — [GitHub Dhruuv26](https://github.com/Dhruuv26/network-intrusion-detection-unsw-nb15)
- A UNSW-NB15 full-feature XGBoost model reached 59.16% macro-F1. — [IIETA](https://iieta.org/download/file/fid/197013)
- NF-UNSW-NB15-v2 multi-class XGBoost: accuracy 0.99 and macro-F1 0.71. — [UNSW-NB15 benchmark summary, Emergent Mind](https://www.emergentmind.com/topics/unsw-nb15)
- A two-stage weighted LightGBM reached a best macro-F1 of 0.670 on UNSW-NB15, but the variant was chosen on the test set, so the figure may be optimistic. — [GitHub sheikhhossainn/UNSW-NB15-GNN](https://github.com/sheikhhossainn/UNSW-NB15-GNN)
- CIC-IDS2017 with a temporal (non-random) split: LSTM had the best macro-F1 at 0.776 and RF was 0.769. Random splits with repeat-last padding can overestimate robustness by up to 0.24 macro-F1. — [Do Transformers Actually Help Intrusion Detection?, arXiv 2606.11098](https://arxiv.org/html/2606.11098v2)
- CIC-IDS2017 with a random split: RF 93.4% macro-F1, rising to about 95.3% with ADASYN. — [arXiv 2105.04301](https://arxiv.org/pdf/2105.04301)
- CICIoT2023: RF was best among RF, Extra Trees, XGBoost, LightGBM, MLP and 1D-CNN, with accuracy 0.9961, macro-F1 0.8698 and weighted-F1 0.9960. The 34-class setup was not fully confirmed. — [CEUR-WS Vol-4281 paper01](https://ceur-ws.org/Vol-4281/paper01.pdf)
- Edge-IIoTset 15-class: RF macro-F1 0.9253 at 98.30% accuracy, and XGB F1 0.9149 (averaging type unclear). RF's per-class F1 for XSS was 0.8246. — [LLM-based multi-class attack analysis, arXiv 2510.26941](https://arxiv.org/pdf/2510.26941)
- Engelen, Rimmer and Joosen (IEEE SPW/WTMC 2021) found problems in CICIDS2017 traffic generation, flow construction (CICFlowMeter TCP termination via FIN/RST), feature extraction and labelling. More than 20% of the traces were reconstructed or relabelled, and ML benchmarks on the corrected data showed significant improvements. — [KU Leuven Lirias PDF](https://lirias.kuleuven.be/retrieve/713070); [code: GintsEngelen/WTMC2021-Code](https://gh.r-mj.com/GintsEngelen/WTMC2021-Code)
- Arp et al. (USENIX Security 2022) describe ten pitfalls in ML for security, the first being sampling bias, where the collected data does not represent the true distribution. They found the pitfalls widespread in 30 top-venue papers. Other pitfalls in the paper include data snooping and inappropriate baselines and metrics. — [USENIX page](https://www.usenix.org/conference/usenixsecurity22/presentation/arp); [PDF](https://mlsec.tu-berlin.de/docs/2022-sec.pdf); [CACM 2024 version](https://mlsec.org/docs/2024-cacm.pdf)

### Inferences
- Expect a tuned XGBoost, LightGBM or RF on ICS3D-container flows to reach well above 0.43 macro-F1. On other CICFlowMeter-style datasets the floor is about 0.6-0.9. If the GBDT gets more than about 0.9, check for duplicates or leakage (identical flows across splits, identifier-like features such as timestamps, ports or IPs) before claiming SODE-Guard parity.
- Always report macro-F1 together with per-class F1 and support. With test supports as small as 26 (about 20% of 131), per-class F1 has high variance, so use stratified repeated splits or multiple seeds with confidence intervals.
- Remove exact duplicates before splitting. A temporal split, where one is possible, gives more honest numbers (up to 0.24 lower macro-F1 on CIC-IDS2017).

### Gaps
- The original Neto et al. 2023 CICIoT2023 baseline macro-F1 and the original Ferrag et al. 2022 Edge-IIoTset baselines were not retrieved.
- ToN-IoT multi-class macro-F1 baselines were not researched within the tool budget.
- No CatBoost-specific NIDS macro-F1 number was found.
- No quantitative study of duplicate-induced inflation for UNSW-NB15 or CICIDS2017 was retrieved (beyond the 0.24 split-protocol effect).

## Q4. Published information on the ICS3D dataset and results reported on it

### Takeaway
ICS3D ("Integrated Cloud Security 3Datasets") exists as a Kaggle compilation under the account rogernickanaedevha. It appears to bundle a container misuse-detection dataset, Edge-IIoTset and a third component. I found no peer-reviewed paper that describes ICS3D or reports results on it. The container-flow component appears to derive from the Kubernetes misuse-detection dataset of Sever and Dogan (ITU J-FET 2023 / AINA 2024), which has about 10 attack scenarios plus benign flows.

### Cited Findings
- The Kaggle page "Integrated Cloud Security 3Datasets (ICS3D)" (rogernickanaedevha) describes a unified collection across three security domains for multi-platform threat analysis. Its description references a container misuse detection dataset and Edge-IIoTset, and alternate names include "EdgeIIoT Federated learning". — [Kaggle ICS3D](https://www.kaggle.com/datasets/rogernickanaedevha/integrated-cloud-security-3datasets-ics3d) (page content seen only through a search summary; the direct fetch was blocked)
- The same account also hosts "Evasion and Poisoning" datasets built on CICIoT2023 (105 IoT devices, 33 attacks). — [Kaggle rogernickanaedevha/poisoning-i](https://www.kaggle.com/datasets/rogernickanaedevha/poisoning-i)
- The Kubernetes misuse-detection dataset by Yigit Sever and Adnan Harun Dogan (ITU Journal on Future and Evolving Technologies 4(2):383-388, 2023) is a public network-flow benchmark created because public container-security datasets are sparse. Its Kaggle version, "Misuse Detection in Containers Dataset (AINA 2024)", lists 10 attack scenarios plus benign flows. — [ITU J-FET article](https://www.itu.int/pub/S-JNL-VOL4.ISSUE2-2023-A26); [PDF](https://www.itu.int/dms_pub/itu-s/opb/jnl/S-JNL-VOL4.ISSUE2-2023-A26-PDF-E.pdf); [Kaggle yigitsever](https://www.kaggle.com/datasets/yigitsever/misuse-detection-in-containers-dataset)
- A separate container-attack dataset (METU thesis, 2022) was built from 2019-2022 CVEs simulated on vulnerable images in Kubernetes. — [METU open archive](https://open.metu.edu.tr/bitstream/handle/11511/97882/10469365.pdf)
- Edge-IIoTset (Ferrag et al. 2022) is an IoT/IIoT testbed dataset for centralized and federated IDS; see Q3 for its baselines. — [Kaggle ICS3D search summary](https://www.kaggle.com/datasets/rogernickanaedevha/integrated-cloud-security-3datasets-ics3d)

### Inferences
- The SODE-Guard container split (9 classes, 131 to about 50k flows, 75 features) is probably a processed version of the Sever and Dogan Kubernetes flows, possibly with some attack scenarios merged or dropped (10 scenarios plus benign would be 11 classes). Check this against the Kaggle file before citing it, and cite the original Sever and Dogan paper for the data's provenance.
- Since no published results on ICS3D were found, the paper must supply its own GBDT/RF/TabM baselines on identical splits. Reviewers will expect these.

### Gaps
- I could not read the Kaggle page directly (kaggle.com was blocked), so the exact three components, class list, per-class counts and licence of ICS3D are unverified.
- No paper, notebook result or leaderboard reporting accuracy or macro-F1 on ICS3D was found.
- Neural SDE intrusion-detection work tied to this dataset was not found.

## Q5. Hybrids: stacking a neural model with a GBDT or kNN, and whether they beat either alone

### Takeaway
Stacking (RF/CatBoost/XGBoost bases with an MLP or LR meta-learner) is common in NIDS and usually improves headline metrics. However, the evidence that it beats the best single GBDT on macro-F1 specifically is weak: one comparison found LightGBM alone more often had the top macro-F1, and the stack was mainly more stable under distribution shift. In the tabular-DL literature, the gains come from ensembling within the NN (TabM) rather than from retrieval or kNN hybrids.

### Cited Findings
- Alsaffar et al. (Journal of Big Data 2024) use RF, CatBoost and XGBoost base learners with an MLP meta-learner and report multi-class gains on UNSW-NB15 and CICIDS2017. Macro-F1 was not confirmed from the snippet. — [Springer J Big Data s40537-024-00994-7](https://link.springer.com/article/10.1186/s40537-024-00994-7)
- A multi-stacking ensemble with dimensionality reduction reports outperforming prior models on CICIDS2017 and UNSW-NB15 across several metrics. — [KAB library record](https://library.kab.ac.ug/Record/doaj-art-3f3e134720e84ce8a09a7c416da52990/Description)
- An XGBoost+LightGBM stack with an LR meta-learner (student thesis, not peer-reviewed) was more consistent under distribution shift, but LightGBM alone more often had the top macro-F1, and the stack was more expensive to compute. — [UPN Jatim repository](https://repository.upnjatim.ac.id/56181/12/22081010194-bab5.pdf)
- TabM reports that retrieval-based tabular methods do not pay off relative to MLP-based efficient ensembles. — [TabM ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)

### Inferences
- A pragmatic hybrid is to average or blend SODE-Guard's calibrated probabilities with a GBDT, then apply post-hoc per-class logit adjustment tuned on validation. This is cheap and lets SODE-Guard contribute uncertainty or temporal-dynamics signal. Report it separately from the pure neural model so that reviewers can see whether the SDE adds value beyond the GBDT.
- Using GBDT leaf indices or predictions as input features to the neural model is another common hybrid, but no NIDS macro-F1 evidence for it was collected here.

### Gaps
- I found no peer-reviewed NIDS study that reports macro-F1 for (neural alone) vs (GBDT alone) vs (neural+GBDT stack) under one protocol.
- No kNN or TabR hybrid results on NIDS were found.
