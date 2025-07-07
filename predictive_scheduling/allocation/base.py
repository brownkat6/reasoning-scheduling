"""
Base classes for token allocation algorithms.

This module defines the interface and common functionality for all allocation
algorithms used in the predictive scheduling framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """
    Result of a token allocation algorithm.
    
    This class encapsulates the output of allocation algorithms, including
    the allocated budgets, metadata, and performance metrics.
    """
    allocations: List[int]
    total_budget: int
    algorithm_name: str
    metadata: Dict[str, Any]
    execution_time: float = 0.0
    
    def __post_init__(self):
        """Validate allocation result after initialization."""
        if len(self.allocations) == 0:
            raise ValueError("Allocations cannot be empty")
        
        if any(alloc < 0 for alloc in self.allocations):
            raise ValueError("All allocations must be non-negative")
        
        actual_total = sum(self.allocations)
        if abs(actual_total - self.total_budget) > 1e-6:
            logger.warning(
                f"Allocation sum ({actual_total}) does not match total budget ({self.total_budget}). "
                f"Difference: {abs(actual_total - self.total_budget)}"
            )
    
    @property
    def num_queries(self) -> int:
        """Number of queries in the allocation."""
        return len(self.allocations)
    
    @property
    def average_allocation(self) -> float:
        """Average allocation per query."""
        return sum(self.allocations) / len(self.allocations)
    
    @property
    def allocation_variance(self) -> float:
        """Variance in allocations across queries."""
        allocations_array = np.array(self.allocations)
        return np.var(allocations_array)
    
    @property
    def allocation_std(self) -> float:
        """Standard deviation in allocations across queries."""
        return np.sqrt(self.allocation_variance)
    
    def get_allocation_stats(self) -> Dict[str, float]:
        """Get comprehensive allocation statistics."""
        allocations_array = np.array(self.allocations)
        return {
            'total_budget': self.total_budget,
            'num_queries': self.num_queries,
            'average_allocation': self.average_allocation,
            'min_allocation': float(np.min(allocations_array)),
            'max_allocation': float(np.max(allocations_array)),
            'median_allocation': float(np.median(allocations_array)),
            'std_allocation': self.allocation_std,
            'variance_allocation': self.allocation_variance,
        }


class BaseAllocator(ABC):
    """
    Abstract base class for token allocation algorithms.
    
    This class defines the interface that all allocation algorithms must implement
    and provides common functionality for validation and metrics computation.
    """
    
    def __init__(self, window_size: int = 16, min_allocation: int = 16, max_allocation: int = 256):
        """
        Initialize base allocator.
        
        Args:
            window_size: Size of allocation windows (tokens)
            min_allocation: Minimum allocation per query (tokens)
            max_allocation: Maximum allocation per query (tokens)
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if min_allocation <= 0:
            raise ValueError(f"min_allocation must be positive, got {min_allocation}")
        if max_allocation <= min_allocation:
            raise ValueError(f"max_allocation ({max_allocation}) must be greater than min_allocation ({min_allocation})")
        
        self.window_size = window_size
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        
        # Ensure min_allocation is a multiple of window_size
        if min_allocation % window_size != 0:
            self.min_allocation = ((min_allocation // window_size) + 1) * window_size
            logger.warning(
                f"Adjusted min_allocation from {min_allocation} to {self.min_allocation} "
                f"to be a multiple of window_size ({window_size})"
            )
    
    @abstractmethod
    def allocate(self, 
                 predictions: Union[np.ndarray, List[float]], 
                 total_budget: int,
                 **kwargs) -> AllocationResult:
        """
        Allocate token budget across queries.
        
        Args:
            predictions: Predictions for each query (format depends on algorithm)
            total_budget: Total token budget to allocate
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            AllocationResult with allocated budgets and metadata
        """
        pass
    
    def validate_inputs(self, 
                       predictions: Union[np.ndarray, List[float]], 
                       total_budget: int) -> np.ndarray:
        """
        Validate and normalize inputs.
        
        Args:
            predictions: Predictions for each query
            total_budget: Total token budget
            
        Returns:
            Normalized predictions as numpy array
        """
        # Convert to numpy array
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        elif not isinstance(predictions, np.ndarray):
            raise TypeError(f"predictions must be list or numpy array, got {type(predictions)}")
        
        # Validate dimensions
        if predictions.ndim == 0:
            raise ValueError("predictions cannot be scalar")
        
        # Validate budget
        if total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {total_budget}")
        
        num_queries = predictions.shape[0]
        min_required_budget = num_queries * self.min_allocation
        
        if total_budget < min_required_budget:
            raise ValueError(
                f"total_budget ({total_budget}) is insufficient for {num_queries} queries. "
                f"Minimum required: {min_required_budget} (min_allocation={self.min_allocation})"
            )
        
        return predictions
    
    def _initialize_allocations(self, num_queries: int) -> List[int]:
        """Initialize allocations with minimum budget for each query."""
        return [self.min_allocation] * num_queries
    
    def _compute_remaining_budget(self, allocations: List[int], total_budget: int) -> int:
        """Compute remaining budget after allocations."""
        return total_budget - sum(allocations)
    
    def _can_allocate_window(self, current_allocation: int) -> bool:
        """Check if another window can be allocated to a query."""
        return current_allocation + self.window_size <= self.max_allocation
    
    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """Get algorithm configuration."""
        return {
            'algorithm_name': self.get_algorithm_name(),
            'window_size': self.window_size,
            'min_allocation': self.min_allocation,
            'max_allocation': self.max_allocation,
        }


class MetricsTracker:
    """Utility class for tracking allocation algorithm performance."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.allocation_history = []
        self.performance_metrics = {}
    
    def add_allocation(self, result: AllocationResult):
        """Add allocation result to history."""
        self.allocation_history.append(result)
    
    def compute_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics across all tracked allocations."""
        if not self.allocation_history:
            return {}
        
        # Extract key metrics
        total_budgets = [r.total_budget for r in self.allocation_history]
        execution_times = [r.execution_time for r in self.allocation_history]
        num_queries = [r.num_queries for r in self.allocation_history]
        
        return {
            'num_allocations': len(self.allocation_history),
            'total_budget_stats': {
                'mean': np.mean(total_budgets),
                'std': np.std(total_budgets),
                'min': np.min(total_budgets),
                'max': np.max(total_budgets),
            },
            'execution_time_stats': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times),
            },
            'num_queries_stats': {
                'mean': np.mean(num_queries),
                'std': np.std(num_queries),
                'min': np.min(num_queries),
                'max': np.max(num_queries),
            }
        }
    
    def get_algorithm_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare performance across different algorithms."""
        if not self.allocation_history:
            return {}
        
        # Group by algorithm
        by_algorithm = {}
        for result in self.allocation_history:
            algo_name = result.algorithm_name
            if algo_name not in by_algorithm:
                by_algorithm[algo_name] = []
            by_algorithm[algo_name].append(result)
        
        # Compute metrics for each algorithm
        comparison = {}
        for algo_name, results in by_algorithm.items():
            execution_times = [r.execution_time for r in results]
            total_budgets = [r.total_budget for r in results]
            
            comparison[algo_name] = {
                'num_runs': len(results),
                'avg_execution_time': np.mean(execution_times),
                'avg_total_budget': np.mean(total_budgets),
                'execution_time_std': np.std(execution_times),
            }
        
        return comparison