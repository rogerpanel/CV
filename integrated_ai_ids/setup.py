"""
Setup script for Integrated AI-IDS
===================================

Installation: pip install -e .

Author: Roger Nick Anaedevha
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="integrated-ai-ids",
    version="1.0.0",
    author="Roger Nick Anaedevha",
    author_email="ar006@campus.mephi.ru",
    description="Integrated AI-powered Intrusion Detection System - PhD Dissertation Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rogerpanel/CV",
    packages=find_packages(exclude=["tests", "docs", "demos"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "cuda": [
            "torch>=2.0.0+cu118",
            "torchaudio>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-ids=integrated_ai_ids.cli:main",
            "ai-ids-server=integrated_ai_ids.api.rest_server:main",
            "ai-ids-demo=integrated_ai_ids.demos.end_to_end_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "integrated_ai_ids": [
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    zip_safe=False,
)
