"""
Simple configuration management for RSI trading system.
"""

from typing import Dict, Any
from pathlib import Path
import json


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for the RSI scanner."""
    return {
        # Entry conditions
        'oversold_threshold': 30.0,
        'exit_threshold': 50.0,
        'rsi_period': 14,

        # Risk management
        'position_size_pct': 1.0,
        'max_hold_days': 60,
        'slippage_pct': 0.001,
        'commission_per_share': 0.01,

        # Filters
        'require_uptrend': True,
        'min_volume': 1000000,
        'enable_zscore_filter': False,
        'zscore_oversold_threshold': -2.0,
        'zscore_window': 50,

        # Advanced features
        'enable_trailing_stop': False,
        'trailing_stop_pct': 5.0,
        'enable_risk_management': False,
        'max_drawdown_pct': 0.20,
        'risk_reduction_factor': 0.5,

        # Statistical validation
        'enable_statistical_validation': True,

        # RSI safeguards
        'disable_rsi_in_trends': True,

        # Out-of-sample testing
        'enable_out_of_sample': False,

        # Data and processing
        'save_results': True,
        'use_risk_adjusted_sizing': False,
        'target_portfolio_volatility': 0.15,
        'max_position_size_pct': 5.0,
        'min_position_size_pct': 0.5,
    }


def load_user_config() -> Dict[str, Any]:
    """Load user configuration from file."""
    config_path = Path("user_config.json")
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge with defaults
            config = get_default_config()
            config.update(user_config)
            return config
        except Exception:
            pass
    return get_default_config()


def save_user_config(config: Dict[str, Any]) -> None:
    """Save user configuration to file."""
    config_path = Path("user_config.json")
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass


def create_config_from_presets(preset_name: str) -> Dict[str, Any]:
    """Create configuration from named presets."""
    presets = get_preset_configs()
    if preset_name in presets:
        config = get_default_config()
        config.update(presets[preset_name])
        return config
    return get_default_config()


def get_preset_configs() -> Dict[str, Dict[str, Any]]:
    """Get available preset configurations."""
    return {
        "Conservative": {
            'oversold_threshold': 30.0,
            'exit_threshold': 50.0,
            'position_size_pct': 0.5,
            'max_hold_days': 90,
            'require_uptrend': True,
            'min_volume': 2000000,
        },
        "Moderate": {
            'oversold_threshold': 25.0,
            'exit_threshold': 60.0,
            'position_size_pct': 1.0,
            'max_hold_days': 60,
            'require_uptrend': True,
            'min_volume': 1000000,
        },
        "Aggressive": {
            'oversold_threshold': 20.0,
            'exit_threshold': 70.0,
            'position_size_pct': 2.0,
            'max_hold_days': 30,
            'require_uptrend': False,
            'min_volume': 500000,
            'enable_trailing_stop': True,
        },
        "Scalping": {
            'oversold_threshold': 35.0,
            'exit_threshold': 45.0,
            'position_size_pct': 0.3,
            'max_hold_days': 14,
            'require_uptrend': True,
            'min_volume': 3000000,
        }
    }


def validate_config(config: Dict[str, Any]) -> list[str]:
    """Validate configuration parameters."""
    errors = []

    # Basic range checks
    if not (0 < config.get('oversold_threshold', 30) < 100):
        errors.append("oversold_threshold must be between 0 and 100")

    if not (0 < config.get('exit_threshold', 50) < 100):
        errors.append("exit_threshold must be between 0 and 100")

    if config.get('exit_threshold', 50) <= config.get('oversold_threshold', 30):
        errors.append("exit_threshold must be greater than oversold_threshold")

    if not (0 < config.get('position_size_pct', 1) <= 100):
        errors.append("position_size_pct must be between 0 and 100")

    if not (1 <= config.get('max_hold_days', 60) <= 365):
        errors.append("max_hold_days must be between 1 and 365")

    if not (0 <= config.get('slippage_pct', 0.001) <= 0.1):
        errors.append("slippage_pct must be between 0 and 0.1")

    return errors
