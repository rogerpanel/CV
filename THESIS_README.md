# PhD Thesis: Advanced Machine Learning Approaches for Network Intrusion Detection Systems

**Author:** Roger Nick Anaedevha  
**Institution:** National Research Nuclear University MEPhI (Moscow Engineering Physics Institute)  
**Date:** November 2025

## Thesis Structure

This repository contains a comprehensive PhD dissertation integrating seven research papers into a unified academic work.

### Main File
- `PhD_Thesis_Integrated.tex` - Main LaTeX document

### Chapters Directory (`chapters/`)

**Front Matter:**
- `titlepage.tex` - Title page
- `abstract.tex` - Comprehensive abstract
- `acknowledgements.tex` - Acknowledgements
- `abbreviations.tex` - List of abbreviations

**Main Chapters:**
1. `chapter1_introduction.tex` - Introduction and Problem Statement
2. `chapter2_literature_review.tex` - Literature Review and Theoretical Foundations
3. `chapter3_neural_ode.tex` - Temporal Adaptive Neural ODEs for Real-Time IDS
4. `chapter4_optimal_transport.tex` - Differentially Private Optimal Transport for Multi-Cloud IDS
5. `chapter5_encrypted_traffic.tex` - Hybrid Deep Learning for Encrypted Traffic Analysis
6. `chapter6_federated_learning.tex` - Federated Learning Approaches for Distributed IDS
7. `chapter7_graph_methods.tex` - Graph-Based Methods for Network Security
8. `chapter8_experiments.tex` - Experimental Evaluation and Comparative Analysis
9. `chapter9_conclusions.tex` - Conclusions and Future Research Directions

**Appendices:**
- `appendix_a_mathematical_proofs.tex` - Mathematical Proofs and Derivations
- `appendix_b_implementation_details.tex` - Implementation Details and Hyperparameters
- `appendix_c_additional_results.tex` - Additional Experimental Results

### Bibliography
- `references.bib` - Consolidated bibliography (203 unique citations)
- `references_index.txt` - Index of all citation keys

## Research Papers Integrated

1. **Temporal Adaptive Neural ODEs with Deep Spatio-Temporal Point Processes**
   - Source: `node_v10ca.tex`, `node_v2ca.tex`
   - Key contributions: TA-BN-ODE, multi-scale temporal modeling, Bayesian inference

2. **Differentially Private Optimal Transport for Multi-Cloud Intrusion Detection**
   - Source: `NotP4_v3c.tex`
   - Key contributions: PPFOT-IDS, Byzantine-robust aggregation, privacy guarantees

3. **Hybrid Spatial-Temporal Deep Learning for Encrypted Traffic Analysis**
   - Source: `eT_Paper_v5cg.tex`
   - Key contributions: CNN-LSTM hybrid, transformer architectures, federated learning

4. **Federated Learning with Graph Temporal Dynamics**
   - Source: `fedgtd-v2.ipynb`
   - Key contributions: Knowledge distillation, communication efficiency

5. **Heterogeneous Graph Pooling for Network Security**
   - Source: `hgp-model-v2.ipynb`
   - Key contributions: Graph-based attack detection, attention mechanisms

6. **Additional Implementations**
   - Neural ODE models: `neural-ode-model-v2-upgraded.ipynb`
   - Optimal transport: `ppfot_ids_complete_final.py`
   - Encrypted traffic: `encrypted_traffic_ids/`

## Datasets

**Integrated Cloud Security 3Datasets (ICS3D):**
- Container Security: 697,289 flows from Kubernetes clusters
- IoT/IIoT Security: 4 million records from seven-layer testbed
- Enterprise SOC: 1 million alerts from 6,100 organizations
- **Access:** Kaggle DOI: [10.34740/kaggle/dsv/12483891](https://doi.org/10.34740/kaggle/dsv/12483891)

**Standard Benchmarks:**
- CIC-IDS2018
- UNSW-NB15
- CIC-IoT-2023

## Compilation Instructions

### Requirements
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages: see preamble in `PhD_Thesis_Integrated.tex`

### Build Steps

```bash
# Navigate to thesis directory
cd /home/user/CV

# First compilation
pdflatex PhD_Thesis_Integrated.tex

# Process bibliography
bibtex PhD_Thesis_Integrated

# Final compilations (2 more times for cross-references)
pdflatex PhD_Thesis_Integrated.tex
pdflatex PhD_Thesis_Integrated.tex
```

Alternative using latexmk:
```bash
latexmk -pdf PhD_Thesis_Integrated.tex
```

### Output
- `PhD_Thesis_Integrated.pdf` - Final dissertation document

## Key Contributions

### Theoretical
- TA-BN-ODE stability analysis through Lyapunov theory
- PAC-Bayesian generalization bounds for security-critical ML
- Byzantine-robust convergence guarantees for federated learning
- Privacy-utility trade-off analysis for differential privacy in OT

### Algorithmic
- 97.3% accuracy with 60-90% parameter reduction (Neural ODE)
- 94.2% accuracy with ε=0.85 differential privacy (Optimal Transport)
- 97-99.9% detection on encrypted traffic without decryption
- 87.6% F1-score on zero-shot novel attack detection (LLM integration)
- 73% energy reduction for edge deployment (SNN conversion)

### Empirical
- 18.9 million security records across 3 domains (ICS3D)
- 12.3 million events/second processing throughput
- Sub-100ms detection latency
- 15-21% improvement over baseline methods
- Cross-domain validation (speech, healthcare)

## Repository Structure

```
CV/
├── PhD_Thesis_Integrated.tex          # Main thesis file
├── chapters/                          # All chapter files
│   ├── titlepage.tex
│   ├── abstract.tex
│   ├── chapter1_introduction.tex
│   ├── ... (all 9 chapters)
│   └── appendix_*.tex
├── references.bib                     # Consolidated bibliography
├── references_index.txt              # Citation index
├── node_v10ca.tex                    # Original Neural ODE paper
├── NotP4_v3c.tex                     # Original Optimal Transport paper
├── eT_Paper_v5cg.tex                 # Original Encrypted Traffic paper
├── *.ipynb                           # Jupyter notebooks with implementations
└── README.md                         # This file
```

## Source Code and Models

**GitHub Repository:** [https://github.com/rogerpanel/CV](https://github.com/rogerpanel/CV)

**Key Implementations:**
- Commit `6bb87d4`: Main training pipeline and models
- Commit `10627314`: Neural ODE models
- Commit `54e68df`: HGP models
- Commit `1e8768b`: FedGTD implementation
- Commit `b52f2e7`: PPFOT-IDS implementation
- Commit `41fe744`: Encrypted traffic IDS

## License

Research code: MIT License  
Datasets (ICS3D): CC BY-NC-SA 4.0  

## Citation

If you use this work, please cite:

```bibtex
@phdthesis{anaedevha2025thesis,
  author = {Anaedevha, Roger Nick},
  title = {Advanced Machine Learning Approaches for Network Intrusion Detection Systems},
  school = {National Research Nuclear University MEPhI},
  year = {2025},
  address = {Moscow, Russia}
}
```

## Supervisors

- Professor Alexander Gennadevich Trofimov
- Professor Yuri Vladimirovich Borodachev  
- Artificial Intelligence Research Center, National Research Nuclear University MEPhI

## Contact

Roger Nick Anaedevha  
Email: ar006@campus.mephi.ru / rogernickanaedevha@gmail.com  
Institution: National Research Nuclear University MEPhI, Moscow, Russia

---

**Last Updated:** November 18, 2025  
**Branch:** claude/integrate-thesis-papers-0176gqefBKT8hgnPh6BVLYGa
