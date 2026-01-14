# Citation and Diagram Integration Guide

**Generated:** January 14, 2026
**Purpose:** Fix missing citations and integrate TikZ diagrams into dissertation chapters

---

## Issues Fixed

### 1. Bibliography Configuration ✅
- **Fixed:** Changed from `\bibliography{references}` back to `\input{references.bib}`
- **Reason:** The `references.bib` file uses `\begin{thebibliography}` format, not BibTeX format
- **Result:** Citations now work properly with `\cite{key}` commands matching `\bibitem{key}` entries

### 2. Introduction Chapter ✅
**File:** `chapters/00_introduction.tex`

**Citations Added:**
- Line 5: Added~\cite{sun2021hybrid,zeadally2020design,hong2022multi,scarfone2007guide,verizon2024dbir,buczak2016survey,corona2019adversarial,truex2019demystifying,zhao2019learning,gama2014survey}
- Line 9: Added~\cite{corona2019adversarial,biggio2013evasion,carlini2017towards,madry2018towards,cohen2019certified}
- Line 11: Added~\cite{zhao2019privacy,mcmahan2017communication,fang2020local,dwork2006calibrating,abadi2016deep,papernot2018sok}
- Line 13: Added~\cite{zhao2019learning,ganin2016domain,zeadally2020design,gama2014survey,kirkpatrick2017overcoming,fang2023llm4ids}
- Method-specific citations added to all 7 contributions (lines 150-175)

**Diagram Added:**
- Problem hierarchy diagram inserted after "Unified Integration" section (line 142)
- Includes both `\input{figures/problem_hierarchy}` and commented `\includegraphics{}` placeholder

### 3. Chapter 2 - Partial ✅
**File:** `chapters/02_mathematical_models.tex`

**Diagrams Added:**
1. ✅ CT-TGNN architecture (after line 9)
2. ✅ TripleE-TGNN architecture (after line 137)
3. ⏳ FedLLM-API architecture (needs adding at line ~232)
4. ⏳ PQ-IDPS architecture (needs adding)
5. ⏳ MambaShield architecture (needs adding)
6. ⏳ Stochastic Transformer architecture (needs adding)
7. ⏳ Game-theoretic architecture (needs adding)

---

## Remaining Tasks

### Task 1: Complete Chapter 2 Diagrams
Add the remaining 5 architecture diagrams to their respective sections:

#### A. FedLLM-API Section (Line ~232)
```latex
\section{Federated Learning with Large Language Models}
\label{sec:models_fedllm}

[Introductory paragraph about federated learning...]

% FedLLM-API Architecture Diagram
\input{figures/fedllm_api_architecture}
% Alternative: \includegraphics[width=0.95\textwidth]{figures/fedllm_api_architecture.pdf}

\subsection{Byzantine-Robust Aggregation}
```

#### B. PQ-IDPS Section (Find line with `\section{Post-Quantum Intrusion Detection}`)
```latex
\section{Post-Quantum Intrusion Detection}
\label{sec:models_pqidps}

[Introductory paragraph...]

% PQ-IDPS Architecture Diagram
\input{figures/pq_idps_architecture}
% Alternative: \includegraphics[width=0.95\textwidth]{figures/pq_idps_architecture.pdf}

\subsection{...}
```

#### C. MambaShield Section
```latex
\section{MambaShield: Adversarially Resilient State Space Models}
\label{sec:models_mambashield}

[Introductory paragraph...]

% MambaShield Architecture Diagram
\input{figures/mambashield_architecture}
% Alternative: \includegraphics[width=0.95\textwidth]{figures/mambashield_architecture.pdf}

\subsection{...}
```

#### D. Stochastic Transformer Section
```latex
\section{Stochastic Transformer with Bayesian Uncertainty Quantification}
\label{sec:models_stochastic}

[Introductory paragraph...]

% Stochastic Transformer Architecture Diagram
\input{figures/stochastic_transformer_architecture}
% Alternative: \includegraphics[width=0.95\textwidth]{figures/stochastic_transformer_architecture.pdf}

\subsection{...}
```

#### E. Game-Theoretic Section
```latex
\section{Game-Theoretic Evasion Resistance}
\label{sec:models_gametheory}

[Introductory paragraph...]

% Game-Theoretic Framework Architecture Diagram
\input{figures/game_theoretic_architecture}
% Alternative: \includegraphics[width=0.95\textwidth]{figures/game_theoretic_architecture.pdf}

\subsection{...}
```

### Task 2: Add Citations to Chapter 2
The Mathematical Models chapter needs ~50-80 citations for:
- Neural ODE papers: \cite{chen2018neural,salvi2024tabn,dupont2019augmented}
- Optimal transport: \cite{cuturi2013sinkhorn,courty2017optimal,damodaran2018deepjdot}
- Federated learning: \cite{mcmahan2017communication,blanchard2017machine}
- Post-quantum crypto: \cite{avanzi2019crystals,bernstein2017post}
- State space models: \cite{gu2024mamba,gu2022efficiently}
- Bayesian deep learning: \cite{blundell2015weight,gal2016dropout,kendall2017uncertainties}
- Game theory: \cite{nash1950equilibrium,bruckner2011stackelberg}

**Key locations to add citations:**
- Section introductions (first paragraph of each section)
- When mentioning specific algorithms or methods
- Theoretical foundations and prior work

### Task 3: Add Citations to Literature Review (Chapter 1)
**File:** `chapters/01_literature_review.tex`

The old `chapter3_literature_review.tex` (60KB) has comprehensive citations. Extract citation patterns from there:

**Example citations needed:**
- Section 1.1: Adversarial ML - \cite{goodfellow2015explaining,carlini2017towards,madry2018towards}
- Section 1.2: Graph NNs - \cite{kipf2017semi,velickovic2018graph,hu2020heterogeneous}
- Section 1.3: Federated Learning - \cite{mcmahan2017communication,kairouz2021advances}
- Section 1.4: Post-Quantum - \cite{bernstein2017post,alagic2020nist}
- Section 1.5: State Space Models - \cite{gu2022efficiently,gu2024mamba}
- Section 1.6: Bayesian DL - \cite{blundell2015weight,gal2016dropout}
- Section 1.7: Game Theory - \cite{bruckner2011stackelberg,liu2017decision}

### Task 4: Add Citations to Implementation (Chapter 3)
**File:** `chapters/03_implementation_results.tex`

Add citations for:
- Datasets: \cite{sharafaldin2018cicids,moustafa2015unsw}
- Evaluation metrics: \cite{powers2011evaluation,naeini2015obtaining}
- Baseline methods being compared against
- Tools and frameworks used (PyTorch, NumPy, etc.)

### Task 5: Add Citations to Conclusion (Chapter 4)
**File:** `chapters/04_conclusion.tex`

Add citations for:
- Future work directions mentioned
- Related work in conclusion
- Broader impact citations

---

## How to Add Remaining Citations

### Method 1: Manual Addition
1. Open each chapter file in a text editor
2. Search for concepts/methods mentioned
3. Add appropriate `\cite{key}` commands from the reference list below
4. Verify citation keys exist in `references.bib`

### Method 2: Extract from Old Dissertation
The old file `chapters/chapter3_literature_review.tex` has ~100+ proper citations.

**To extract citation patterns:**
```bash
# Get all citations from old literature review
grep -o '\\cite{[^}]*}' chapters/chapter3_literature_review.tex | sort -u > old_citations.txt

# View with context
grep -B2 -A2 '\\cite{' chapters/chapter3_literature_review.tex > citations_with_context.txt
```

---

## Available Citation Keys

Based on `references.bib`, available citation keys include:

### Core ML/DL
- chen2018neural (Neural ODEs)
- salvi2024tabn (Temporal Adaptive BN)
- dupont2019augmented (Augmented ODEs)
- kingma2014auto (Variational autoencoders)
- blundell2015weight (Bayes by Backprop)
- gal2016dropout (MC Dropout)
- kendall2017uncertainties (Uncertainty types)

### Adversarial ML
- goodfellow2015explaining (FGSM)
- madry2018towards (PGD)
- carlini2017towards (C&W attack)
- corona2019adversarial (Adversarial IDS)
- cohen2019certified (Certified robustness)
- biggio2013evasion (Evasion attacks)

### Federated Learning
- mcmahan2017communication (FedAvg)
- kairouz2021advances (FL advances)
- blanchard2017machine (Byzantine-robust)
- fang2020local (Poisoning attacks)
- abadi2016deep (DP-SGD)

### Graph Neural Networks
- kipf2017semi (GCN)
- velickovic2018graph (GAT)
- hu2020heterogeneous (HGT)
- hamilton2017inductive (GraphSAGE)

### Optimal Transport
- cuturi2013sinkhorn (Sinkhorn)
- courty2017optimal (OT for DA)
- damodaran2018deepjdot (DeepJDOT)

### Post-Quantum Cryptography
- bernstein2017post (PQC survey)
- alagic2020nist (NIST PQC)
- avanzi2019crystals (CRYSTALS-Kyber)

### State Space Models
- gu2022efficiently (S4)
- gu2024mamba (Mamba)

### Game Theory
- nash1950equilibrium (Nash equilibrium)
- bruckner2011stackelberg (Stackelberg games)
- liu2017decision (Security games)
- littlestone1994weighted (Weighted majority)

### Intrusion Detection
- scarfone2007guide (IDS guide)
- buczak2016survey (ML for IDS)
- verizon2024dbir (Data breach report)

### Privacy
- dwork2006calibrating (Differential privacy)
- abadi2016deep (DP-SGD)
- papernot2018sok (Privacy in ML)

### Transformers & LLMs
- vaswani2017attention (Attention is all you need)
- devlin2019bert (BERT)
- brown2020language (GPT-3)
- wei2022chain (Chain-of-thought)

### Miscellaneous
- gama2014survey (Concept drift)
- guo2017calibration (Calibration)
- kirkpatrick2017overcoming (EWC)

---

## Diagram Compilation Options

### Option 1: Use TikZ Directly (Current Default)
- Diagrams compile as vector graphics
- High quality, scalable
- Requires TikZ packages
- **Current setup:** `\input{figures/diagram_name}`

### Option 2: Pre-compile to PDF/PNG
If TikZ compilation is slow or problematic:

1. **Compile each diagram separately:**
```bash
cd figures
for file in *_architecture.tex problem_hierarchy.tex; do
    pdflatex -jobname=$(basename $file .tex) <<EOF
\documentclass{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usetikzlibrary{positioning,arrows,shapes,calc,patterns}
\input{$file}
\bye
EOF
done
```

2. **Convert to PNG if needed:**
```bash
for file in *.pdf; do
    convert -density 300 $file -quality 90 ${file%.pdf}.png
done
```

3. **Uncomment the `\includegraphics{}` lines** in chapters and comment out the `\input{}` lines

---

## Testing Compilation

### Quick Test
```bash
cd Dissertation_Reorganized
pdflatex PhD_Dissertation_Reorganized.tex
```

### Full Compilation (with bibliography)
```bash
pdflatex PhD_Dissertation_Reorganized.tex
# Note: Since references.bib uses \begin{thebibliography}, NO bibtex step needed
pdflatex PhD_Dissertation_Reorganized.tex
pdflatex PhD_Dissertation_Reorganized.tex
```

### Expected Warnings
- Some overfull hbox warnings (normal for long equations)
- Possible figure placement warnings
- Missing citations warnings until all citations are added

### Citation Verification
After adding citations, check for undefined references:
```bash
grep "Citation.*undefined" PhD_Dissertation_Reorganized.log
```

---

## Summary of Current Status

### ✅ Completed
1. Fixed bibliography configuration
2. Added ~30 citations to Introduction
3. Added problem hierarchy diagram to Introduction
4. Added 2/7 architecture diagrams to Chapter 2 (CT-TGNN, TripleE-TGNN)
5. Added citation framework to Introduction contributions

### ⏳ In Progress
1. Adding remaining 5 architecture diagrams to Chapter 2
2. Adding comprehensive citations to Chapter 2

### 📝 Pending
1. Citations for Literature Review (Chapter 1)
2. Citations for Implementation (Chapter 3)
3. Citations for Conclusion (Chapter 4)
4. Final compilation test
5. Verification that all citations resolve correctly

---

## Next Steps Recommendation

1. **Immediate:** Complete adding 5 remaining diagrams to Chapter 2 (10 minutes)
2. **High Priority:** Add citations throughout Chapter 2 Mathematical Models (30-60 minutes)
3. **Medium Priority:** Add citations to Literature Review (30 minutes)
4. **Low Priority:** Add citations to Implementation and Conclusion (20 minutes each)
5. **Final:** Test full compilation and fix any issues (15 minutes)

**Total estimated time to complete:** 2-3 hours

---

## Contact for Issues

If you encounter:
- **Undefined citation warnings:** Check that citation key exists in `references.bib`
- **TikZ compilation errors:** Try pre-compiling diagrams to PDF
- **Figure placement issues:** Adjust `[htbp]` placement specifiers
- **Bibliography not showing:** Verify `\input{references.bib}` is correct

The current setup should work correctly once all citations and diagrams are added.
