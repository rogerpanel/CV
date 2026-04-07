"""
Configuration management for encrypted traffic IDS experiments

Supports YAML configuration files with dot-notation access
and automatic directory creation for logging.
"""

import yaml
import os
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """
    Configuration object with dot-notation access.

    Example:
        >>> config = Config({'model': {'d_model': 512}})
        >>> config.model.d_model  # 512
    """

    def __init__(self, data: dict = None):
        if data:
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(path: str = 'configs/config.yaml') -> Config:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Config object with dot-notation access
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # Validate required sections
    required = ['data', 'model', 'training']
    for section in required:
        if section not in data:
            raise ValueError(f"Missing required config section: {section}")

    # Create logging directories
    log_dir = data.get('logging', {}).get('log_dir', './logs')
    ckpt_dir = data.get('logging', {}).get('checkpoint_dir', './checkpoints')
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    return Config(data)


def save_config(config: Config, path: str) -> None:
    """Save configuration to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to: {path}")


def update_config(config: Config, key: str, value: Any) -> None:
    """
    Update configuration value using dot-notation key.

    Example:
        >>> update_config(config, 'training.learning_rate', 0.0001)
    """
    keys = key.split('.')
    obj = config
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], value)
