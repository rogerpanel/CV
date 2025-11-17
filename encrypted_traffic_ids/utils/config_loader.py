"""
Configuration loader for encrypted traffic IDS experiments

This module handles loading and validation of experiment configurations
from YAML files, ensuring all required parameters are present.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os


class Config:
    """
    Configuration object with dot notation access.

    Allows accessing nested dictionary values using dot notation,
    e.g., config.model.cnn_lstm.lstm_hidden_dim
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.
                    If None, loads default config.yaml from configs directory.

    Returns:
        Config: Configuration object with dot notation access

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is not valid YAML

    Example:
        >>> config = load_config('configs/config.yaml')
        >>> print(config.training.learning_rate)
        0.001
    """
    if config_path is None:
        # Default to configs/config.yaml in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'configs' / 'config.yaml'

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    print(f"Loading configuration from: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Validate required sections
    required_sections = ['data', 'model', 'training']
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required configuration section: {section}")

    # Create output directories if specified
    if 'logging' in config_dict:
        log_dir = config_dict['logging'].get('log_dir', './logs')
        checkpoint_dir = config_dict['logging'].get('checkpoint_dir', './checkpoints')

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    return Config(config_dict)


def save_config(config: Config, save_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object
        save_path: Path to save YAML file

    Example:
        >>> config = load_config()
        >>> save_config(config, 'outputs/experiment_config.yaml')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)

    print(f"Configuration saved to: {save_path}")


def update_config(config: Config, updates: Dict[str, Any]) -> Config:
    """
    Update configuration with new values.

    Args:
        config: Original configuration
        updates: Dictionary of updates (supports nested keys with dots)

    Returns:
        Config: Updated configuration

    Example:
        >>> config = load_config()
        >>> config = update_config(config, {'training.learning_rate': 0.0001})
    """
    config_dict = config.to_dict()

    for key, value in updates.items():
        # Support nested keys with dot notation
        keys = key.split('.')
        current = config_dict

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    return Config(config_dict)
