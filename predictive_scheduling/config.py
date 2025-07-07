"""
Configuration management for Predictive Scheduling framework.

This module provides centralized configuration management with environment-based
settings, proper validation, and secure handling of sensitive data.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml


logger = logging.getLogger(__name__)


@dataclass
class PathConfig:
    """Configuration for data and model paths."""
    base_path: str = ""
    data_stem: str = ""
    x_stem: str = ""
    y_stem: str = ""
    models_stem: str = ""
    
    def __post_init__(self):
        """Validate and normalize paths after initialization."""
        self.base_path = Path(self.base_path).expanduser().resolve()
        self.data_stem = Path(self.data_stem).expanduser().resolve()
        self.x_stem = Path(self.x_stem).expanduser().resolve()
        self.y_stem = Path(self.y_stem).expanduser().resolve()
        self.models_stem = Path(self.models_stem).expanduser().resolve()


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    hidden_size: int = 1536
    num_layers: int = 28
    cache_dir: Optional[str] = None
    device: str = "auto"
    use_flash_attention: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    weight_decay: float = 0.0
    dropout: float = 0.0
    seed: int = 42
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"


@dataclass
class AllocationConfig:
    """Configuration for token allocation algorithms."""
    window_size: int = 16
    min_allocation: int = 16
    max_allocation: int = 256
    probe_suffix: str = "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\\n\\n\\[ \\boxed{"


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    
    # Environment and user settings
    user: str = field(default_factory=lambda: os.environ.get('USER', ''))
    environment: str = field(default_factory=lambda: os.environ.get('ENVIRONMENT', 'development'))
    debug: bool = field(default_factory=lambda: os.environ.get('DEBUG', 'false').lower() == 'true')
    
    def __post_init__(self):
        """Initialize paths based on user and environment."""
        if not self.paths.base_path:
            self._set_default_paths()
    
    def _set_default_paths(self):
        """Set default paths based on user configuration."""
        if self.user == "katrinabrown":
            base_path = "/n/netscratch/dwork_lab/Lab/katrina"
            self.paths.base_path = base_path
            self.paths.data_stem = f"{base_path}/reasoning_scheduling_new/"
            self.paths.x_stem = f"{base_path}/reasoning_scheduling_new/data/"
            self.paths.y_stem = f"{base_path}/reasoning_scheduling_new/data/"
            self.paths.models_stem = f"{base_path}/models/"
        elif self.user == "amuppidi":
            base_path = "/n/netscratch/gershman_lab/Lab/amuppidi"
            self.paths.base_path = base_path
            self.paths.data_stem = f"{base_path}/reasoning_scheduling_new/"
            self.paths.x_stem = f"{base_path}/reasoning_scheduling_new_orig/data/"
            self.paths.y_stem = f"{base_path}/reasoning_scheduling_new/data/"
            self.paths.models_stem = f"{base_path}/models/"
        else:
            # Default to current directory for unknown users
            current_dir = Path.cwd()
            self.paths.base_path = current_dir
            self.paths.data_stem = current_dir / "data"
            self.paths.x_stem = current_dir / "data"
            self.paths.y_stem = current_dir / "data"
            self.paths.models_stem = current_dir / "models"
            
            logger.warning(
                f"Unknown user '{self.user}'. Using default paths in current directory. "
                "Consider setting paths explicitly via configuration file."
            )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        paths_dict = config_dict.pop('paths', {})
        model_dict = config_dict.pop('model', {})
        training_dict = config_dict.pop('training', {})
        lora_dict = config_dict.pop('lora', {})
        allocation_dict = config_dict.pop('allocation', {})
        
        return cls(
            paths=PathConfig(**paths_dict),
            model=ModelConfig(**model_dict),
            training=TrainingConfig(**training_dict),
            lora=LoRAConfig(**lora_dict),
            allocation=AllocationConfig(**allocation_dict),
            **config_dict
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'paths': self.paths.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'lora': self.lora.__dict__,
            'allocation': self.allocation.__dict__,
            'user': self.user,
            'environment': self.environment,
            'debug': self.debug
        }
    
    def save_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self):
        """Validate configuration parameters."""
        errors = []
        
        # Validate paths exist or can be created
        for path_name, path_value in self.paths.__dict__.items():
            if path_name.endswith('_stem') or path_name == 'models_stem':
                path_obj = Path(path_value)
                if not path_obj.exists():
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created directory: {path_obj}")
                    except PermissionError:
                        errors.append(f"Cannot create directory: {path_obj}")
        
        # Validate model parameters
        if self.model.hidden_size <= 0:
            errors.append("Model hidden_size must be positive")
        
        if self.model.num_layers <= 0:
            errors.append("Model num_layers must be positive")
        
        # Validate training parameters
        if self.training.batch_size <= 0:
            errors.append("Training batch_size must be positive")
        
        if self.training.learning_rate <= 0:
            errors.append("Training learning_rate must be positive")
        
        if self.training.num_epochs <= 0:
            errors.append("Training num_epochs must be positive")
        
        # Validate LoRA parameters
        if self.lora.rank <= 0:
            errors.append("LoRA rank must be positive")
        
        if self.lora.alpha <= 0:
            errors.append("LoRA alpha must be positive")
        
        # Validate allocation parameters
        if self.allocation.window_size <= 0:
            errors.append("Allocation window_size must be positive")
        
        if self.allocation.min_allocation <= 0:
            errors.append("Allocation min_allocation must be positive")
        
        if self.allocation.max_allocation <= self.allocation.min_allocation:
            errors.append("Allocation max_allocation must be greater than min_allocation")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
        
        logger.info("Configuration validation passed")


def get_default_config() -> Config:
    """Get default configuration based on environment."""
    return Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or create default.
    
    Args:
        config_path: Path to configuration file. If None, looks for default locations.
    
    Returns:
        Loaded configuration object.
    """
    if config_path is None:
        # Look for config in standard locations
        possible_paths = [
            "config.yaml",
            "configs/config.yaml", 
            "~/.predictive_scheduling/config.yaml",
            "/etc/predictive_scheduling/config.yaml"
        ]
        
        for path in possible_paths:
            path_obj = Path(path).expanduser()
            if path_obj.exists():
                config_path = str(path_obj)
                break
    
    if config_path and Path(config_path).exists():
        logger.info(f"Loading configuration from: {config_path}")
        config = Config.from_yaml(config_path)
    else:
        logger.info("No configuration file found, using default configuration")
        config = get_default_config()
    
    config.validate()
    return config