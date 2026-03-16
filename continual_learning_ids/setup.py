"""Package setup for CL-RL NIDS framework."""

from setuptools import setup, find_packages
from pathlib import Path

readme = Path(__file__).parent.parent / "continual_learning_ids" / "README.md"
long_description = readme.read_text() if readme.exists() else ""

setup(
    name="continual-learning-ids",
    version="1.0.0",
    description=(
        "Continual Learning and Constrained Reinforcement Learning for "
        "Adversarially Robust Network Intrusion Detection and Autonomous Response"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Roger Nick Anaedevha, Alexander G. Trofimov, Yuri V. Borodachev",
    author_email="roger@robustidps.ai",
    url="https://github.com/rogerpanel/CV/tree/main/continual_learning_ids",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking :: Monitoring",
    ],
    keywords=[
        "intrusion-detection",
        "continual-learning",
        "reinforcement-learning",
        "constrained-mdp",
        "ewc",
        "cpo",
        "adversarial-robustness",
        "network-security",
    ],
)
