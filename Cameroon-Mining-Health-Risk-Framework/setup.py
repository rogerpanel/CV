"""Editable install for the Cameroon-Mining-Health-Risk-Framework (`cmhr`) package."""
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
LONG_DESC = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="cmhr",
    version="1.0.0",
    description=(
        "Reproducible framework for environmental quality assessment and probabilistic "
        "health-risk modelling in artisanal mining landscapes (East Cameroon)."
    ),
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="Roger Nick Anaedevha",
    license="MIT",
    url="https://github.com/rogerpanel/CV",
    package_dir={"cmhr": "src"},
    packages=["cmhr"] + ["cmhr." + p for p in find_packages(where="src")],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.1",
        "scipy>=1.11",
        "scikit-learn>=1.4",
        "pyyaml>=6.0",
        "tqdm>=4.66",
    ],
    extras_require={
        "ml": ["xgboost>=2.0", "lightgbm>=4.1", "shap>=0.44"],
        "geo": ["geopandas>=0.14", "rasterio>=1.3", "libpysal>=4.9", "esda>=2.5", "pykrige>=1.7"],
        "gee": ["earthengine-api>=0.1.380", "geemap>=0.30"],
        "viz": ["matplotlib>=3.8", "seaborn>=0.13", "folium>=0.15"],
        "all": [
            "xgboost>=2.0", "lightgbm>=4.1", "shap>=0.44",
            "geopandas>=0.14", "rasterio>=1.3", "libpysal>=4.9", "esda>=2.5", "pykrige>=1.7",
            "earthengine-api>=0.1.380", "geemap>=0.30",
            "matplotlib>=3.8", "seaborn>=0.13", "folium>=0.15", "SALib>=1.4",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
)
