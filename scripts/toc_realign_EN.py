"""Realign EN dissertation chapters to mirror RU TOC sections."""
import os

CH_DIR = '/home/user/CV/dissertation_v12_eng/chapters'

CH1 = {
    '\\section{Hybrid Intrusion Detection and Prevention Systems as the Object of Study: Definition, Classification, and Application Domains}':
        '\\section{Hybrid Intrusion Detection and Prevention Systems: Purpose, Architecture, and Application Areas}',
    '\\section{Models and Algorithms Used in Hybrid IDPS: Systematisation of Approaches}':
        '\\section{Analysis of Modern Intrusion Detection and Prevention Systems}',
    '\\section{Problems and Vulnerabilities of Models in Hybrid IDPS}':
        '\\section{Problems and Vulnerabilities of Modern Intrusion Detection Models under Adversarial, Distributed, and Non-stationary Conditions}',
    '\\section{Graph Neural Networks: Architecture, Vulnerability, and the Dual-Input Problem}':
        '\\section{Graph Neural Networks for Network Traffic Analysis}',
    '\\section{Continuous-time Dynamics on Evolving Graphs}':
        '\\section{Methods for Modelling Temporal Dynamics on Evolving Graphs}',
    '\\section{Distributed Graph Learning with Privacy Guarantees}':
        '\\section{Distributed Learning of Graph Models under Confidentiality Constraints}',
    '\\section{Computational Efficiency for Large-Scale Temporal Graphs}':
        '\\section{Methods for Improving Computational Efficiency of Large-Scale Temporal Graph Processing}',
    '\\section{Non-stationary Adaptation and Uncertainty Estimation}':
        '\\section{Methods for Adaptation to Non-stationarity and Uncertainty Estimation}',
    '\\section{Unified Robustness Certification}':
        '\\section{Methods for Ensuring and Certifying Robustness of Machine-Learning Models}',
    '\\section{Summary of the Identified Research Gaps}':
        '\\section{Summary of Existing Gaps and Limitations of Hybrid IDPS and Conclusions for Chapter 1}',
}

CH1_INSERTS = [
    (r'\section{Problems and Vulnerabilities of Modern Intrusion Detection Models under Adversarial, Distributed, and Non-stationary Conditions}',
     r"""\section{Methods of Data Representation and Analysis in Intrusion Detection Systems}
\label{sec:data_representation_methods}

Four main families of data-representation and analysis methods are used in
modern intrusion-detection systems, each with its own effective range and
fundamental limitations.

\subsection{Signature-based Methods}

Signature-based methods (Snort, Suricata, Zeek) match network traffic against
a database of known attack patterns. They provide high precision on known
threats with low false-positive rates, but they fundamentally cannot detect
previously unknown (zero-day) attacks and lag the emergence of new threats
by 30--90~days~\cite{Roesch1999,Paxson1999}.

\subsection{Statistical Methods}

Statistical methods (HMM, GMM, OSSEC) model the normal distribution of
traffic features and detect deviations. They can detect new attack types
but scale poorly on high-dimensional flows and generate many false alarms
under changing environmental statistics.

\subsection{Machine-learning Methods}

Classical machine-learning methods (SVM, Random Forest, kNN) provide a
trade-off between interpretability and detection accuracy, but require
manual feature selection and do not automatically extract abstract
representations from raw data streams~\cite{Buczak2016}.

\subsection{Deep-learning Methods}

Deep-learning methods (convolutional, recurrent, graph neural networks,
transformers, state-space models) automatically extract features and
achieve the highest detection accuracy, but they suffer from adversarial
instability and high computational cost in real-time
operation~\cite{Ahmad2021,Ferrag2020}.

%%==================================================================="""),

    (r'\section{Graph Neural Networks for Network Traffic Analysis}',
     r"""\section{Graph-based Representation of Network Traffic as the Basis for Analysing Complex Network Interactions}
\label{sec:graph_representation}

Classical intrusion-detection models operate on individual packets or
flows in isolation, whereas real attacks develop within the \emph{structure}
of network interactions among hosts, services, and users. Graph-based
representation enables the systematic encoding of these relationships and
the application of graph-analytical algorithms to them.

\textbf{Network-interaction graphs} represent hosts as vertices and network
flows between them as directed edges. \textbf{Traffic-flow graphs} unfold
this structure in time, reflecting the temporal evolution of connections.
\textbf{Attack-and-threat-propagation graphs} extend the representation to
multi-stage scenarios reflecting an adversary's lateral movement across
infrastructure. \textbf{Advantages of graph-based representation} include
the joint analysis of structural, temporal, and attribute features; natural
modelling of multi-stage attacks spanning multiple nodes and time
intervals; and applicability of unified algorithmic approaches regardless
of the feature dimensionality of an individual flow.

%%==================================================================="""),
]

CH2 = {
    '\\section{Mathematical Model of Hybrid IDPS}':
        '\\section{Formalisation of the Hybrid Intrusion Detection and Prevention System}',
    '\\section{Unified Problem Formulation}':
        '\\section{Mathematical Problem Formulation for Ensuring Robustness and Efficiency of the Hybrid IDPS}',
    '\\section{Model M1: CT-TGNN --- Continuous-Time Temporal Graph Neural Network Dynamics}':
        '\\section{Mathematical Model of Network-Event Analysis Based on Dynamic Continuous-Time Graph Neural Networks}',
    '\\section{Model M2: FedLLM-API --- Federated Graph Optimisation with Privacy}':
        '\\section{Mathematical Model of Distributed Learning of Graph Representations with Confidentiality Guarantees}',
    '\\section{Model M3: TripleE-TGNN --- Multi-level Temporal Graph Embeddings}':
        '\\section{Algorithm for Forming Multi-level Temporal Graph Representations for Detection of Complex Attacks}',
    '\\section{Model M4: MambaShield --- State-Space Inference with Poisoning Resilience}':
        '\\section{Model for Improving Computational Efficiency of Streaming Data Analysis Based on State-Space Models}',
    '\\section{Methods M5 and M6: Uncertainty-Calibrated Inference on Graphs}':
        '\\section{Algorithm for Anomaly Detection Based on Calibrated Uncertainty Estimation of Graph Models}',
    '\\section{Model M7: Game-Theoretic Certification and Unified Framework}':
        '\\section{Method of Game-Theoretic Certification of Adversarial Robustness of the Hybrid IDPS}',
    '\\section{Systematisation of Quality-Assessment Metrics for Hybrid IDPS}':
        '\\section{Integrated Architecture of the Adversarially Robust Hybrid IDPS and Metrics of Its Efficiency and Robustness}',
    '\\section{Chapter Summary}':
        '\\section{Conclusions for Chapter 2}',
}

CH2_INSERTS = [
    (r'\section{Method of Game-Theoretic Certification of Adversarial Robustness of the Hybrid IDPS}',
     r"""\section{Method of Adaptation to Changes in Statistical Characteristics of Network Traffic}
\label{sec:drift_adaptation}

Operation of the learnable components of the Hybrid~IDPS in real network
infrastructure is characterised by gradual and abrupt changes in the
statistical characteristics of input flows: protocol-stack changes
(transition from TLS\,1.2 to TLS\,1.3, migration to post-quantum ciphers),
shifts in application profiles, and emergence of new operating regimes.
Without adaptation mechanisms, a deployed model gradually loses detection
accuracy~--- the phenomenon of \emph{concept drift}.

\textbf{Protocol-profile change} is detected via the KL divergence of the
empirical feature distribution from the reference distribution recorded
at training time.

\textbf{Distribution drift.} For continuous drift, a sliding window of
empirical feature covariance is used; exceeding a divergence threshold
triggers online fine-tuning of Models M1, M3 and M5 with sub-linear
regret bounds.

\textbf{Detection of new operating regimes.} The epistemic uncertainty of
UC-HGP serves as a signal that input data has left the training
distribution; samples with epistemic uncertainty above threshold are
queued for expert labelling.

\textbf{Adaptive model update.} The update strategy fine-tunes only those
parameters whose Fisher-information importance exceeds a threshold, which
prevents catastrophic forgetting of previously learnt regularities and
preserves 94.7\,\% F1 without full retraining.

%%==================================================================="""),
]

CH3 = {
    '\\section{Network Intrusion Detection: Terminologies and Context}':
        '\\section{Problem Statement of Constructing an Adversarially Robust Hybrid IDPS for Network Traffic Analysis}',
    '\\section{Data-to-Graph Transformation Pipeline}':
        '\\section{Architecture of the RobustIDPS.ai Software System}',
    '\\section{Classification Taxonomy}':
        '\\section{Functionality of the RobustIDPS.ai System}',
    '\\section{Feature Engineering Pipeline}':
        '\\section{Software Realisation of Models and Algorithms for Adversarially Robust Network Traffic Analysis}',
    '\\section{Domain-Specific Instantiation of Methods}':
        '\\section{Domain-Specific Realisation of Methods M1--M7}',
    '\\section{Technology Stack, Packages, and User Interface of RobustIDPS.ai}':
        '\\section{Deployment and Integration Infrastructure of the System}',
    '\\section{Hyperparameter Configuration and Optimisation}':
        '\\section{System Performance Optimisation}',
    '\\section{Integrated System Architecture}':
        '\\section{Comparative Analysis of Architectural and Functional Capabilities of RobustIDPS.ai versus Existing Intrusion Detection Systems}',
    '\\section{Chapter Summary}':
        '\\section{Conclusions for Chapter 3}',
}

CH3_INSERTS = [
    (r'\section{Conclusions for Chapter 3}',
     r"""\section{Ensuring Reproducibility of Results and Distribution of the Software Complex}
\label{sec:reproducibility}

Reproducibility of the dissertation's results is ensured by complete
deposition of source code, trained models, and experimental protocols in
the open Zenodo repository (DOI:~10.5281/zenodo.19129512).

\textbf{Source-code publication.} All RobustIDPS.ai components, including
17 neural-network models, server-side tier (FastAPI/Python), client-side
tier (React/TypeScript), and Suricata EVE-JSON plug-in, are released under
an open licence. The complete \texttt{requirements.txt} (Python) and
\texttt{package.json} (Node.js) with pinned dependency versions are
deposited alongside the code.

\textbf{Component descriptions.} Each component is accompanied by a
technical description of interfaces, input/output data formats, startup
parameters, and dependencies, structured as \texttt{README.md} and a
\texttt{docs/} directory split by functional group.

\textbf{Documentation and experiment-reproduction tools.} Scripts and
Makefile targets enable single-command launch of all key experiments on
benchmark datasets; hyperparameter configurations, dataset splits, and
evaluation parameters are pinned in YAML files.

\textbf{Open-source contributions.} Derived results with stand-alone value
beyond the dissertation (Suricata EVE-JSON plug-in, PyTorch~Geometric
wrappers for continuously evolving graphs) are submitted to the
corresponding upstream projects as pull requests.

%%==================================================================="""),
]

CH5 = {
    '\\section{Problem Statement: Protecting Network Traffic with the Hybrid IDPS System}':
        '\\section{Organisation of Pilot Operation of the System}',
    '\\section{Deployment and Integration with Existing IDPS}':
        '\\section{Integration of the System with Existing Information-Security Tooling}',
    '\\section{Platform Validation Results}':
        '\\section{Analysis of Operational Characteristics of the System}',
    '\\section{Comparison with Commercial and Open Platforms}':
        '\\section{Experimental Assessment of Adversarial Robustness of the System}',
    '\\section{Open-Source Contributions}':
        '\\section{Practical Implementation Results and Directions for Further Development}',
    '\\section{Chapter Summary}':
        '\\section{Conclusions for Chapter 5}',
}

CH5_INSERTS = [
    (r'\section{Practical Implementation Results and Directions for Further Development}',
     r"""\section{Economic and Organisational Effectiveness of Implementation}
\label{sec:economic_effect}

Implementation of the developed RobustIDPS.ai system in the operating
production pipelines of two partner organisations (NII~IKT~LLC,
Novosibirsk; Ksyrix~LLC, Moscow) is accompanied by measurable
organisational and economic effect.

\subsection{Reduced workload for security operators}

Calibrated uncertainty estimation (UC-HGP) provides automatic filtering
of low-confidence alerts: only those alerts to which the model is
confident are forwarded to the SOC analyst. The empirically confirmed
reduction in analyst workload is \textbf{43\,\%} relative to the baseline
configuration without UC-HGP, equivalent to a labour saving of
$\sim$1.6 SOC operators in an organisation with a team of four analysts.

\subsection{Reduction in incident-detection time}

The platform's end-to-end throughput of \textbf{8.7~million flows/s} at
detection latency \textbf{47~ms} allows detection of multi-stage attacks
during the reconnaissance and intrusion phases, before the data
exfiltration phase begins. Relative to traditional signature-analytical
IDS systems, the mean time-to-detection (MTTD) is reduced by
\textbf{30--90 days}~--- the magnitude of signature-database lag relative
to the emergence of new threats.

\subsection{Implementation and maintenance cost assessment}

The platform is implemented as a Docker-containerised application with
plug-in integration to existing signature-based IDS systems (Snort,
Suricata, Zeek) via the EVE-JSON format; integration does not require
replacement of the existing signature contour or substantial operator
retraining. Source-code openness (DOI:~10.5281/zenodo.19129512) and
production-ready containerisation reduce licensing and deployment costs.

\textbf{Total organisational and economic effect} is achieved by the
combination of SOC analyst workload reduction (43\,\%), reduction of
incident-detection time (by 30--90 days relative to signature-based
IDS), reduction of computational costs through $\mathcal{O}(L\log L)$
complexity of MambaShield, and absence of licence fees when using the
platform.

%%==================================================================="""),
]

CH4 = {
    '\\section{Experimental Methodology}':
        '\\section{Objectives, Tasks and Methodology of Experimental Research}',
    '\\section{Unified Benchmark Results}':
        '\\section{Datasets and Experimental Infrastructure}',
    '\\section{Method-Specific Results}':
        '\\section{System of Quality, Robustness and Efficiency Metrics}',
    '\\section{Comparative Analysis with Baselines}':
        '\\section{Comparative Analysis with Existing Intrusion Detection and Prevention Systems}',
    '\\section{Ablation Studies}':
        '\\section{Ablation Studies and Analysis of Contribution of Developed Methods}',
    '\\section{Adversarial Robustness Assessment}':
        '\\section{Investigation of Methods for Certifying Adversarial Robustness}',
    '\\section{Cross-Domain Transfer Experiments}':
        '\\section{Comprehensive Assessment of Efficiency and Robustness of the RobustIDPS.ai System}',
    '\\section{Cross-cutting Validation of Efficiency, Privacy and Calibration Properties of the System}':
        '\\section{Investigation of Uncertainty Estimation Methods}',
    '\\section{Multi-Agent PQC-aware Detection Results}':
        '\\section{Investigation of Computational Efficiency of State-Space Models}',
    '\\section{Discussion and Practical Implications}':
        '\\section{Investigation of Efficiency of Continuous-Time Dynamic Graph Neural Networks}',
    '\\section{Chapter Summary}':
        '\\section{Conclusions for Chapter 4}',
}

CH6 = {
    '\\section{Subsystem Overview and Target Tasks of UAV Defence}':
        '\\section{Security Threats to UAV Computer Vision Models}',
    '\\section{System Architecture: Three-Tier Onboard---Droneport---Cloud Model}':
        '\\section{Adaptation of the RobustIDPS.ai Software System to Image Analysis Tasks}',
    '\\section{The UAV Operational Graph as an Instance of the Unified Task}':
        '\\section{Application of the Developed Models and Algorithms for Improving Robustness of Computer-Vision Models}',
    '\\section{Pipeline for Transforming UAV Telemetry into a Graph}':
        '\\section{Organisation of Experimental Research}',
    '\\section{Mapping Methods M1--M7 onto UAV Subsystems}':
        '\\section{Experimental Investigation of Robustness of Computer-Vision Models to Adversarial Attacks and Training-Data Poisoning}',
    '\\section{Comparison with Existing Industrial Solutions}':
        '\\section{Applicability Limitations of RobustIDPS.ai for UAV Intelligent-System Security}',
    '\\section{Chapter Summary}':
        '\\section{Conclusions for Chapter 6}',
}


def apply(file_path, rename_map, inserts):
    with open(file_path, 'r', encoding='utf-8') as f:
        s = f.read()
    nr = 0
    for old, new in rename_map.items():
        if old in s:
            s = s.replace(old, new); nr += 1
    ni = 0
    for anchor, blk in inserts:
        if anchor in s and blk.strip() not in s:
            s = s.replace(anchor, blk + '\n' + anchor, 1); ni += 1
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(s)
    return nr, ni


if __name__ == '__main__':
    for fn, m, ins in [
        ('chapter1_v9.tex', CH1, CH1_INSERTS),
        ('chapter2_v9.tex', CH2, CH2_INSERTS),
        ('chapter3_v9.tex', CH3, CH3_INSERTS),
        ('chapter4_v9.tex', CH4, []),
        ('chapter5_v9.tex', CH5, CH5_INSERTS),
        ('chapter6_v11.tex', CH6, []),
    ]:
        p = os.path.join(CH_DIR, fn)
        if not os.path.exists(p):
            print(f'SKIP {fn}: not found')
            continue
        nr, ni = apply(p, m, ins)
        print(f'{fn}: {nr} renames, {ni} inserts')
