"""Configuration utilities.

Load and merge YAML configuration files with inheritance support.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


def load_config(config_path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file with inheritance.

    Supports '_base_' key for inheriting from parent configs.

    Args:
        config_path: Path to YAML config file.
        base_dir: Base directory for resolving relative paths.

    Returns:
        Configuration dictionary.

    Example:
        Config file with inheritance:
        ```yaml
        _base_: '../base.yaml'

        model:
          variant: 'm'
        ```
    """
    config_path = Path(config_path)

    if base_dir is None:
        base_dir = config_path.parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Handle inheritance
    if '_base_' in config:
        base_path = Path(base_dir) / config.pop('_base_')
        base_config = load_config(str(base_path), base_dir=base_path.parent)
        config = merge_configs(base_config, config)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Values in override take precedence. Nested dicts are merged recursively.

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        Merged configuration.
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def config_to_namespace(config: Dict[str, Any]) -> 'Namespace':
    """Convert config dict to namespace for attribute access.

    Args:
        config: Configuration dictionary.

    Returns:
        Namespace object with nested namespaces for nested dicts.
    """
    class Namespace:
        def __init__(self, d: Dict[str, Any]):
            for key, value in d.items():
                if isinstance(value, dict):
                    setattr(self, key, Namespace(value))
                else:
                    setattr(self, key, value)

        def __repr__(self):
            return f"Namespace({self.__dict__})"

        def to_dict(self) -> Dict[str, Any]:
            result = {}
            for key, value in self.__dict__.items():
                if isinstance(value, Namespace):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
            return result

    return Namespace(config)


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """Pretty print configuration.

    Args:
        config: Configuration dictionary.
        indent: Indentation level.
    """
    for key, value in config.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")
