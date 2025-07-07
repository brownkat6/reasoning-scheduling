"""
Predictive Scheduling for Efficient Inference-Time Reasoning in Large Language Models.

This package implements the predictive scheduling framework described in the paper:
"Predictive Scheduling for Efficient Inference-Time Reasoning in Large Language Models"

The framework enables dynamic token budget allocation based on:
1. MLP predictors trained on transformer hidden states
2. LoRA fine-tuned models for difficulty classification  
3. Greedy allocation algorithms for optimal resource distribution

Key modules:
- config: Configuration management
- models: MLP and LoRA model implementations
- data: Data processing and loading utilities
- allocation: Token budget allocation algorithms
- evaluation: Model evaluation and metrics
- training: Training loops and utilities
"""

__version__ = "1.0.0"
__author__ = "Aneesh Muppidi, Katrina Brown, Michael Mitzenmacher"
__email__ = "aneeshmuppidi@college.harvard.edu"

from .config import Config, load_config, get_default_config

__all__ = [
    "Config",
    "load_config", 
    "get_default_config",
]