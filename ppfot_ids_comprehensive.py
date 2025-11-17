#!/usr/bin/env python3
"""
PPFOT-IDS: Privacy-Preserving Federated Optimal Transport Intrusion Detection System
Complete Implementation for Q1 Journal Publication

Paper: Differentially Private Optimal Transport for Multi-Cloud Intrusion Detection
Author: Roger Nick Anaedevha
Corresponding LaTeX: NotP4_v3c.tex

This script implements all methodological components, evaluation metrics, and experiments
described in the paper for reproducible research.
"""

print("="*80)
print("PPFOT-IDS: Complete Implementation")
print("="*80)
print("\nInitializing modules...")

# Standard library imports
import os
import sys
import time
import warnings
import pickle
import json
from collections import defaultdict, OrderedDict
from typing import Tuple, List, Dict, Optional, Union
from datetime import datetime

# Scientific computing
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.autograd import Variable

# Scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)

# Optimal transport
try:
    import ot  # Python Optimal Transport (POT)
    import geomloss
    print("✓ Optimal transport libraries loaded")
except ImportError as e:
    print(f"⚠ Warning: {e}")
    print("  Install with: pip install pot geomloss")

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Data loading
try:
    import kagglehub
    print("✓ Kaggle Hub loaded")
except ImportError:
    print("⚠ Kaggle Hub not available (data must be provided manually)")
    kagglehub = None

# Progress bars
from tqdm import tqdm

# Configure environment
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

print("\n✓ All modules loaded successfully\n")

