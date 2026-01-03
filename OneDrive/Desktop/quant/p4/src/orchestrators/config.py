"""
Configuration orchestrator - manages user configuration and presets.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Simplified config - using utils.config now
from src.utils.errors import ConfigError


def load_user_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load user configuration from file with environment variable overrides.

    Args:
        config_path: Path to config file (optional)

    Returns:
        Dictionary with merged configuration
    """
    # Default config path
    if config_path is None:
        config_path = Path("config.yaml")

    # Load base config from file
    base_config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                base_config = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigError(f"Failed to load config from {config_path}: {e}")

    # Apply environment variable overrides
    config = apply_env_overrides(base_config)

    # Validate configuration
    validate_config(config)

    return config


def apply_env_overrides(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to base config.

    Args:
        base_config: Base configuration dictionary

    Returns:
        Configuration with environment overrides applied
    """
    config = base_config.copy()

    # Environment variable mappings
    env_mappings = {
        'RSI_OVERSOLD_THRESHOLD': ('oversold_threshold', float),
        'RSI_EXIT_THRESHOLD': ('exit_threshold', float),
        'POSITION_SIZE_PCT': ('position_size_pct', float),
        'SLIPPAGE_PCT': ('slippage_pct', float),
        'COMMISSION_PER_SHARE': ('commission_per_share', float),
        'MAX_HOLD_DAYS': ('max_hold_days', int),
        'REQUIRE_UPTREND': ('require_uptrend', lambda x: x.lower() == 'true'),
        'ENABLE_ZSCORE_FILTER': ('enable_zscore_filter', lambda x: x.lower() == 'true'),
        'MIN_VOLUME': ('min_volume', int),
        'SAVE_RESULTS': ('save_results', lambda x: x.lower() == 'true'),
    }

    for env_var, (config_key, converter) in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                config[config_key] = converter(env_value)
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid value for {env_var}={env_value}: {e}")

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ConfigError: If configuration is invalid
    """
    try:
        # Try to create Pydantic models to validate
        BacktestConfig(**config)
        ScannerConfig(**config)
    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {e}")


def save_user_config(config: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        config_path: Path to save config (optional)
    """
    if config_path is None:
        config_path = Path("config.yaml")

    try:
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise ConfigError(f"Failed to save config to {config_path}: {e}")


def get_preset_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get predefined configuration presets.

    Returns:
        Dictionary of preset names to configurations
    """
    return {
        'conservative': {
            'oversold_threshold': 25,
            'exit_threshold': 60,
            'position_size_pct': 1.0,
            'slippage_pct': 0.001,
            'commission_per_share': 0.01,
            'max_hold_days': 90,
            'require_uptrend': True,
            'enable_zscore_filter': False,
            'min_volume': 2000000,
        },

        'aggressive': {
            'oversold_threshold': 35,
            'exit_threshold': 45,
            'position_size_pct': 3.0,
            'slippage_pct': 0.002,
            'commission_per_share': 0.01,
            'max_hold_days': 30,
            'require_uptrend': False,
            'enable_zscore_filter': True,
            'zscore_oversold_threshold': -1.5,
            'min_volume': 500000,
        },

        'default': get_default_config(),
    }


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Dictionary with default configuration
    """
    return {
        # Entry conditions
        'oversold_threshold': 30,
        'require_uptrend': True,

        # Exit conditions
        'exit_threshold': 50,
        'max_hold_days': 60,
        'enable_trailing_stop': False,
        'trailing_stop_pct': 5.0,

        # Filters
        'enable_zscore_filter': False,
        'zscore_oversold_threshold': -2.0,
        'min_volume': 1000000,

        # Risk management
        'position_size_pct': 2.0,
        'slippage_pct': 0.001,
        'commission_per_share': 0.01,

        # Indicators
        'rsi_period': 14,
        'zscore_window': 50,

        # Other
        'save_results': True,
    }


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.

    Args:
        base_config: Base configuration
        override_config: Configuration to override with

    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged


def create_config_from_presets(preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create configuration from preset with optional overrides.

    Args:
        preset_name: Name of preset configuration
        overrides: Optional configuration overrides

    Returns:
        Configuration dictionary

    Raises:
        ConfigError: If preset doesn't exist
    """
    presets = get_preset_configs()

    if preset_name not in presets:
        available = list(presets.keys())
        raise ConfigError(f"Preset '{preset_name}' not found. Available: {available}")

    config = presets[preset_name].copy()

    if overrides:
        config.update(overrides)

    return config


def get_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get human-readable summary of configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with configuration summary
    """
    return {
        'entry_conditions': {
            'rsi_oversold_threshold': config.get('oversold_threshold'),
            'require_uptrend': config.get('require_uptrend'),
            'zscore_filter_enabled': config.get('enable_zscore_filter'),
            'min_volume': config.get('min_volume'),
        },
        'exit_conditions': {
            'rsi_exit_threshold': config.get('exit_threshold'),
            'max_hold_days': config.get('max_hold_days'),
            'trailing_stop_enabled': config.get('enable_trailing_stop'),
        },
        'risk_management': {
            'position_size_pct': config.get('position_size_pct'),
            'slippage_pct': config.get('slippage_pct') * 100,  # Convert to percentage
            'commission_per_share': config.get('commission_per_share'),
        },
        'indicators': {
            'rsi_period': config.get('rsi_period'),
            'zscore_window': config.get('zscore_window'),
        },
    }
