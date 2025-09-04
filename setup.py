#!/usr/bin/env python3
"""
Sequential Pattern Transformer (SPT) - Setup Script
A PyTorch-based framework for sequence pattern analysis using transformer models.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sequential-pattern-transformer",
    version="1.0.0",
    author="Your Name",  # TODO: Replace with your actual name
    author_email="your.email@example.com",  # TODO: Replace with your email
    description="A PyTorch-based framework for sequence pattern analysis using transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SPT",  # TODO: Replace with your GitHub URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinxcontrib-bibtex>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spt-train=spt.main:main",
            "spt-evaluate=spt.evaluation.model_evaluation:main",
            "spt-analyze=spt.analysis.disease_relationship_analysis.encoder_relationship_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "spt": [
            "data/*.txt",
            "data/*.pkl",
            "data/*.csv",
            "analysis/disease_relationship_analysis/*.csv",
            "analysis/disease_relationship_analysis/*.pth",
            "analysis/disease_relationship_analysis/*.pkl",
        ],
    },
    zip_safe=False,
    keywords="transformer, sequence-analysis, medical-ai, pytorch, nlp, healthcare",
)
