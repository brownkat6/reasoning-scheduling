"""
Setup script for Predictive Scheduling framework.

This package implements the predictive scheduling framework described in:
"Predictive Scheduling for Efficient Inference-Time Reasoning in Large Language Models"
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Get version
def get_version():
    """Get version from __init__.py."""
    init_file = os.path.join("predictive_scheduling", "__init__.py")
    with open(init_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split('"')[1]
    return "0.1.0"

setup(
    name="predictive-scheduling",
    version=get_version(),
    author="Aneesh Muppidi, Katrina Brown, Michael Mitzenmacher",
    author_email="aneeshmuppidi@college.harvard.edu",
    description="Predictive Scheduling for Efficient Inference-Time Reasoning in Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brownkat6/reasoning-scheduling",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
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
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.21.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "peft>=0.4.0",
        "openai>=1.0.0",
        "rich>=12.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.60.0",
        "datasets>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "isort>=5.10",
            "flake8>=4.0",
            "mypy>=0.950",
            "pre-commit>=2.15",
        ],
        "wandb": [
            "wandb>=0.13.0",
        ],
        "security": [
            "keyring>=23.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "all": [
            "wandb>=0.13.0",
            "keyring>=23.0.0", 
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "predictive-scheduling=predictive_scheduling.cli:main",
            "dynasor-chat=predictive_scheduling.dynasor.client:main",
        ],
    },
    include_package_data=True,
    package_data={
        "predictive_scheduling": ["config.yaml", "*.yaml", "*.yml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/brownkat6/reasoning-scheduling/issues",
        "Source": "https://github.com/brownkat6/reasoning-scheduling",
        "Documentation": "https://github.com/brownkat6/reasoning-scheduling/blob/main/README.md",
        "Paper": "https://arxiv.org/abs/2024.xxxxx",  # Update with actual arXiv link
    },
    keywords=[
        "machine learning",
        "natural language processing", 
        "language models",
        "inference optimization",
        "token scheduling",
        "adaptive computation",
        "early stopping",
        "predictive scheduling",
    ],
    zip_safe=False,
)