"""
Token allocation algorithms for predictive scheduling.

This module implements the core allocation algorithms that distribute token budgets
across queries based on predicted early stopping probabilities or difficulty classifications.
"""

from .greedy import GreedyAllocator, DifficultyBasedAllocator
from .oracle import OracleAllocator
from .baseline import UniformAllocator
from .base import BaseAllocator, AllocationResult
from .utils import validate_allocation, compute_allocation_metrics

__all__ = [
    "GreedyAllocator",
    "DifficultyBasedAllocator", 
    "OracleAllocator",
    "UniformAllocator",
    "BaseAllocator",
    "AllocationResult",
    "validate_allocation",
    "compute_allocation_metrics",
]