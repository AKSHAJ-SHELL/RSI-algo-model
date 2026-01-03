"""
Consolidated trading strategies for RSI mean-reversion system.

Combines all strategy implementations into a single module for easier maintenance.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    VOLATILE = "volatile"
    CALM = "calm"


# Base Strategy Class
class BaseStrategy(ABC):
    """Abstract base class for all RSI-based trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for display."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of the strategy."""
        pass

    @property
    @abstractmethod
    def risk_level(self) -> str:
        """Risk level: 'Low', 'Medium', or 'High'."""
        pass

    @property
    @abstractmethod
    def market_fit(self) -> str:
        """Market conditions where this strategy performs best."""
        pass

    @abstractmethod
    def get_entry_description(self) -> str:
        """Describe the entry conditions in plain English."""
        pass

    @abstractmethod
    def get_exit_description(self) -> str:
        """Describe the exit conditions in plain English."""
        pass

    @abstractmethod
    def get_educational_explanation(self) -> str:
        """Educational explanation for beginners."""
        pass

    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get the valid parameter ranges for this strategy."""
        pass

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameter values for this strategy."""
        pass

    @abstractmethod
    def generate_entry_signals(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Generate entry signals for this strategy."""
        pass

    @abstractmethod
    def generate_exit_signals(self, data: pd.DataFrame, entry_signals: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Generate exit signals for this strategy."""
        pass

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information for UI display."""
        return {
            'name': self.name,
            'description': self.description,
            'risk_level': self.risk_level,
            'market_fit': self.market_fit,
            'entry_description': self.get_entry_description(),
            'exit_description': self.get_exit_description(),
            'educational_explanation': self.get_educational_explanation(),
            'parameter_ranges': self.get_parameter_ranges(),
            'default_parameters': self.get_default_parameters(),
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate strategy parameters against allowed ranges."""
        errors = []
        ranges = self.get_parameter_ranges()

        for param_name, (min_val, max_val) in ranges.items():
            if param_name in parameters:
                value = parameters[param_name]
                if not isinstance(value, (int, float)):
                    errors.append(f"{param_name} must be a number")
                elif value < min_val or value > max_val:
                    errors.append(f"{param_name} must be between {min_val} and {max_val}")

        return errors

    def get_risk_color(self) -> str:
        """Get color code for risk level visualization."""
        if self.risk_level == "Low":
            return "🟢"  # Green
        elif self.risk_level == "Medium":
            return "🟡"  # Yellow
        elif self.risk_level == "High":
            return "🔴"  # Red
        return "⚪"  # White (unknown)

    def get_complexity_level(self) -> str:
        """Get complexity level based on parameters and logic."""
        num_params = len(self.get_parameter_ranges())
        if num_params <= 3:
            return "Beginner"
        elif num_params <= 6:
            return "Intermediate"
        else:
            return "Advanced"

    def get_required_indicators(self) -> List[str]:
        """Get list of required indicators for this strategy."""
        rsi_period = self.get_default_parameters().get('rsi_period', 14)
        return [f"RSI_{rsi_period}"]

    def get_adaptive_parameters(self, data: pd.DataFrame, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get market regime-adaptive parameters."""
        try:
            from .indicators import detect_market_regime, get_regime_adaptive_parameters
            regime = detect_market_regime(data, lookback=60)
            adaptive_config = get_regime_adaptive_parameters(base_config, regime)
            return adaptive_config
        except Exception as e:
            return base_config


# Concrete Strategy Implementations

class ConservativeMeanReversion(BaseStrategy):
    """RSI Conservative Mean-Reversion Strategy."""

    @property
    def name(self) -> str:
        return "Conservative Mean-Reversion"

    @property
    def description(self) -> str:
        return "Buy when RSI drops below 30, sell when RSI rises above 50. Perfect for beginners in range-bound markets."

    @property
    def risk_level(self) -> str:
        return "Low"

    @property
    def market_fit(self) -> str:
        return "Range-bound markets with moderate volatility"

    def get_entry_description(self) -> str:
        return "Enter long when RSI drops below 30 (oversold condition)"

    def get_exit_description(self) -> str:
        return "Exit when RSI rises above 50 (mean reversion target)"

    def get_educational_explanation(self) -> str:
        return """
        **RSI Conservative Mean-Reversion Strategy**

        **What it does:**
        This strategy waits for stocks to become "oversold" (RSI < 30) and buys, expecting the price to rebound to fair value (RSI > 50).

        **Why it works:**
        RSI measures if a stock is overbought (>70) or oversold (<30). When RSI drops below 30, it suggests the stock is temporarily undervalued and may bounce back.

        **Best for:**
        - Range-bound markets (sideways price action)
        - Beginners learning technical analysis
        - Conservative investors wanting steady, predictable returns

        **Risk considerations:**
        - May miss opportunities in strong uptrends
        - Requires patience during extended downtrends
        - Works best in choppy, sideways markets

        **Example:** If a stock's RSI drops to 25, this strategy would buy, expecting it to rise back toward 50-60.
        """

    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'oversold_threshold': (20.0, 35.0),  # RSI entry threshold
            'exit_threshold': (45.0, 60.0),      # RSI exit threshold
            'rsi_period': (10, 21),              # RSI calculation period
            'max_hold_days': (30, 120),          # Maximum holding period
            'position_size_pct': (0.5, 2.0),     # Position size as % of capital
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'oversold_threshold': 30.0,
            'exit_threshold': 50.0,
            'rsi_period': 14,
            'max_hold_days': 60,
            'position_size_pct': 1.0,
            'slippage_pct': 0.001,
            'commission_per_share': 0.01,
            'require_uptrend': False,
        }

    def generate_entry_signals(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Generate entry signals using RSI oversold conditions."""
        signals = pd.Series(0, index=data.index, name='entry_signal')

        if 'RSI_14' in data.columns:
            rsi_threshold = config.get('oversold_threshold', 30)
            signals[data['RSI_14'] <= rsi_threshold] = 1

        return signals

    def generate_exit_signals(self, data: pd.DataFrame, entry_signals: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Generate exit signals using RSI mean reversion targets."""
        signals = pd.Series(0, index=data.index, name='exit_signal')

        if 'RSI_14' in data.columns:
            rsi_threshold = config.get('exit_threshold', 50)
            signals[data['RSI_14'] >= rsi_threshold] = 1

        # Time-based exit
        max_hold_days = config.get('max_hold_days', 60)
        if max_hold_days > 0:
            entry_dates = entry_signals[entry_signals == 1].index
            for entry_date in entry_dates:
                exit_date = entry_date + pd.Timedelta(days=max_hold_days)
                if exit_date in signals.index:
                    signals.loc[exit_date] = 1

        return signals


class AggressiveMeanReversion(BaseStrategy):
    """RSI Aggressive Mean-Reversion Strategy."""

    @property
    def name(self) -> str:
        return "Aggressive Mean-Reversion"

    @property
    def description(self) -> str:
        return "Buy on extreme oversold conditions, sell on overbought. Higher risk, higher reward for experienced traders."

    @property
    def risk_level(self) -> str:
        return "High"

    @property
    def market_fit(self) -> str:
        return "Range-bound markets with high volatility"

    def get_entry_description(self) -> str:
        return "Enter long when RSI drops below 25 (extreme oversold) with volume confirmation"

    def get_exit_description(self) -> str:
        return "Exit when RSI reaches 70 (overbought) or trailing stop is hit"

    def get_educational_explanation(self) -> str:
        return """
        **RSI Aggressive Mean-Reversion Strategy**

        **What it does:**
        This strategy enters on extreme oversold conditions (RSI < 25) and exits on overbought conditions (RSI > 70), with tighter stops for quicker profits.

        **Why it works:**
        Extreme RSI readings often signal short-term reversals. The aggressive approach captures bigger moves but requires careful risk management.

        **Best for:**
        - Experienced traders comfortable with volatility
        - High-volatility range-bound markets
        - Traders seeking higher returns with higher risk

        **Risk considerations:**
        - More false signals than conservative approach
        - Requires strict risk management
        - Can be whipsawed in choppy markets
        - Higher position sizes increase risk

        **Example:** RSI drops to 20 with high volume - aggressive entry. RSI reaches 75 - aggressive exit.
        """

    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'oversold_threshold': (15.0, 30.0),     # More aggressive entry
            'exit_threshold': (65.0, 80.0),         # Higher exit target
            'rsi_period': (10, 21),
            'max_hold_days': (15, 60),              # Shorter holding period
            'position_size_pct': (1.0, 5.0),        # Larger positions
            'trailing_stop_pct': (3.0, 10.0),       # Tighter stops
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'oversold_threshold': 25.0,
            'exit_threshold': 70.0,
            'rsi_period': 14,
            'max_hold_days': 30,
            'position_size_pct': 2.0,
            'trailing_stop_pct': 5.0,
            'slippage_pct': 0.001,
            'commission_per_share': 0.01,
            'require_uptrend': False,
            'enable_trailing_stop': True,
        }

    def generate_entry_signals(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Generate aggressive entry signals on extreme oversold conditions."""
        signals = pd.Series(0, index=data.index, name='entry_signal')

        if 'RSI_14' in data.columns:
            rsi_threshold = config.get('oversold_threshold', 25)
            # More aggressive entry with volume confirmation
            rsi_condition = data['RSI_14'] <= rsi_threshold
            volume_condition = data['Volume'] > data['Volume'].rolling(20).mean() * 1.2  # 20% above average
            signals[rsi_condition & volume_condition] = 1

        return signals

    def generate_exit_signals(self, data: pd.DataFrame, entry_signals: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Generate aggressive exit signals with multiple conditions."""
        signals = pd.Series(0, index=data.index, name='exit_signal')

        if 'RSI_14' in data.columns:
            rsi_threshold = config.get('exit_threshold', 70)
            signals[data['RSI_14'] >= rsi_threshold] = 1

        # Add trailing stop logic
        if config.get('enable_trailing_stop', False):
            stop_pct = config.get('trailing_stop_pct', 5.0) / 100.0
            # Simplified trailing stop implementation
            in_position = entry_signals.cumsum().astype(bool)
            if in_position.any():
                entry_prices = data.loc[entry_signals == 1, 'Close']
                if not entry_prices.empty:
                    entry_price = entry_prices.iloc[0]
                    trailing_stop = data['Low'] < entry_price * (1 - stop_pct)
                    signals[trailing_stop & in_position] = 1

        return signals


class DivergenceCatcher(BaseStrategy):
    """RSI Divergence Catcher Strategy."""

    @property
    def name(self) -> str:
        return "Divergence Catcher"

    @property
    def description(self) -> str:
        return "Catch RSI divergences for higher-probability reversals. Advanced strategy for experienced traders."

    @property
    def risk_level(self) -> str:
        return "Medium"

    @property
    def market_fit(self) -> str:
        return "Any market with established trends"

    def get_entry_description(self) -> str:
        return "Enter on bullish RSI divergence (price makes lower low, RSI makes higher low)"

    def get_exit_description(self) -> str:
        return "Exit on bearish RSI divergence or RSI reaches 70"

    def get_educational_explanation(self) -> str:
        return """
        **RSI Divergence Catcher Strategy**

        **What it does:**
        This strategy looks for divergences between price and RSI. When price makes a lower low but RSI makes a higher low, it signals weakening downward momentum.

        **Why it works:**
        Divergences indicate a potential reversal. RSI divergences are particularly reliable because RSI is a momentum oscillator.

        **Best for:**
        - Experienced traders who can identify chart patterns
        - Markets with clear trends showing signs of reversal
        - Traders seeking higher-probability setups

        **Risk considerations:**
        - Requires pattern recognition skills
        - Can be subjective in identification
        - False signals in very strong trends
        - Needs confirmation from other indicators

        **Example:** Price drops to new low at $50, but RSI only drops to 35 (previous low was 30). This bullish divergence suggests upward reversal.
        """

    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'rsi_period': (10, 21),
            'divergence_threshold': (5, 20),        # Minimum divergence size
            'max_hold_days': (20, 90),
            'position_size_pct': (0.5, 3.0),
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'rsi_period': 14,
            'divergence_threshold': 10,
            'max_hold_days': 45,
            'position_size_pct': 1.5,
            'slippage_pct': 0.001,
            'commission_per_share': 0.01,
            'require_uptrend': False,
        }

    def generate_entry_signals(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Generate entry signals based on RSI divergence patterns."""
        signals = pd.Series(0, index=data.index, name='entry_signal')

        # Use divergence detection from indicators module
        if 'rsi_divergence' in data.columns:
            signals[data['rsi_divergence'] == 1] = 1  # Bullish divergence

        return signals

    def generate_exit_signals(self, data: pd.DataFrame, entry_signals: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Generate exit signals based on RSI targets or bearish divergence."""
        signals = pd.Series(0, index=data.index, name='exit_signal')

        if 'RSI_14' in data.columns:
            signals[data['RSI_14'] >= 70] = 1  # Overbought exit

        if 'rsi_divergence' in data.columns:
            signals[data['rsi_divergence'] == -1] = 1  # Bearish divergence exit

        return signals


class MultiTimeframeConfirmation(BaseStrategy):
    """Multi-Timeframe RSI Confirmation Strategy."""

    @property
    def name(self) -> str:
        return "Multi-Timeframe Confirmation"

    @property
    def description(self) -> str:
        return "RSI signals confirmed across multiple timeframes for higher accuracy."

    @property
    def risk_level(self) -> str:
        return "Medium"

    @property
    def market_fit(self) -> str:
        return "Range-bound markets with consistent behavior across timeframes"

    def get_entry_description(self) -> str:
        return "Enter when RSI is oversold on multiple timeframes (daily + weekly)"

    def get_exit_description(self) -> str:
        return "Exit when RSI normalizes on primary timeframe"

    def get_educational_explanation(self) -> str:
        return """
        **Multi-Timeframe RSI Confirmation Strategy**

        **What it does:**
        This strategy requires RSI oversold signals to be confirmed across both daily and weekly timeframes, reducing false signals.

        **Why it works:**
        Multi-timeframe confirmation filters out short-term noise and focuses on signals that have broader market significance.

        **Best for:**
        - Traders who can analyze multiple timeframes
        - Markets with consistent behavior across timeframes
        - Traders seeking higher-accuracy signals

        **Risk considerations:**
        - Fewer signals (may miss opportunities)
        - Requires access to multiple timeframes
        - Can be late to enter strong moves

        **Example:** Daily RSI < 30 AND Weekly RSI < 35 = confirmed oversold signal.
        """

    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'daily_oversold_threshold': (20.0, 35.0),
            'weekly_oversold_threshold': (25.0, 40.0),
            'rsi_period': (10, 21),
            'max_hold_days': (30, 90),
            'position_size_pct': (0.5, 2.5),
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'daily_oversold_threshold': 30.0,
            'weekly_oversold_threshold': 35.0,
            'rsi_period': 14,
            'max_hold_days': 60,
            'position_size_pct': 1.5,
            'slippage_pct': 0.001,
            'commission_per_share': 0.01,
            'require_uptrend': True,
        }

    def generate_entry_signals(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Generate entry signals with multi-timeframe confirmation."""
        signals = pd.Series(0, index=data.index, name='entry_signal')

        daily_threshold = config.get('daily_oversold_threshold', 30.0)
        weekly_threshold = config.get('weekly_oversold_threshold', 35.0)

        # Check if we have both daily and weekly RSI
        if 'RSI_14' in data.columns and 'RSI_weekly_14' in data.columns:
            daily_condition = data['RSI_14'] <= daily_threshold
            weekly_condition = data['RSI_weekly_14'] <= weekly_threshold
            signals[daily_condition & weekly_condition] = 1

        return signals

    def generate_exit_signals(self, data: pd.DataFrame, entry_signals: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Generate exit signals based on RSI normalization."""
        signals = pd.Series(0, index=data.index, name='exit_signal')

        if 'RSI_14' in data.columns:
            signals[data['RSI_14'] >= 55] = 1  # Normalized exit

        return signals


class VolatilityFilterStrategy(BaseStrategy):
    """RSI with Volatility Filter Strategy."""

    @property
    def name(self) -> str:
        return "Volatility Filter Strategy"

    @property
    def description(self) -> str:
        return "RSI signals filtered by volatility conditions for optimal market timing."

    @property
    def risk_level(self) -> str:
        return "Medium"

    @property
    def market_fit(self) -> str:
        return "Markets with varying volatility - adapts to current conditions"

    def get_entry_description(self) -> str:
        return "Enter on RSI oversold signals only during appropriate volatility conditions"

    def get_exit_description(self) -> str:
        return "Exit on RSI targets with volatility-adjusted position sizing"

    def get_educational_explanation(self) -> str:
        return """
        **Volatility Filter Strategy**

        **What it does:**
        This strategy combines RSI signals with volatility analysis. It enters during moderate volatility periods and avoids extreme conditions.

        **Why it works:**
        RSI signals work best in certain volatility environments. This strategy adapts position sizes and thresholds based on current market volatility.

        **Best for:**
        - Markets with changing volatility conditions
        - Traders who want adaptive strategies
        - Risk management focused traders

        **Risk considerations:**
        - Complex parameter optimization
        - May miss signals in optimal conditions
        - Requires volatility indicator calculation

        **Example:** High volatility = smaller positions, wider thresholds. Low volatility = larger positions, tighter thresholds.
        """

    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'base_oversold_threshold': (20.0, 35.0),
            'base_exit_threshold': (45.0, 65.0),
            'volatility_lookback': (10, 50),
            'max_position_size_pct': (1.0, 5.0),
            'min_position_size_pct': (0.2, 1.0),
            'rsi_period': (10, 21),
            'max_hold_days': (20, 80),
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'base_oversold_threshold': 30.0,
            'base_exit_threshold': 55.0,
            'volatility_lookback': 20,
            'max_position_size_pct': 3.0,
            'min_position_size_pct': 0.5,
            'rsi_period': 14,
            'max_hold_days': 45,
            'position_size_pct': 1.5,
            'slippage_pct': 0.001,
            'commission_per_share': 0.01,
            'require_uptrend': False,
            'use_risk_adjusted_sizing': True,
        }

    def generate_entry_signals(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Generate entry signals with volatility filtering."""
        signals = pd.Series(0, index=data.index, name='entry_signal')

        if 'RSI_14' in data.columns and 'volatility_thresholds' in data.columns:
            # Use dynamic thresholds from volatility analysis
            oversold_thresholds = data['volatility_thresholds']['oversold']
            signals[data['RSI_14'] <= oversold_thresholds] = 1

        return signals

    def generate_exit_signals(self, data: pd.DataFrame, entry_signals: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Generate exit signals with volatility-adjusted targets."""
        signals = pd.Series(0, index=data.index, name='exit_signal')

        if 'RSI_14' in data.columns:
            if 'volatility_thresholds' in data.columns:
                overbought_thresholds = data['volatility_thresholds']['overbought']
                signals[data['RSI_14'] >= overbought_thresholds] = 1
            else:
                signals[data['RSI_14'] >= 60] = 1  # Default exit

        return signals


# Strategy Registry
class StrategyRegistry:
    """Registry for managing and accessing trading strategies."""

    def __init__(self):
        self._strategies: Dict[str, type] = {}
        self._strategy_instances: Dict[str, BaseStrategy] = {}
        self._strategy_info: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_strategies()

    def _register_builtin_strategies(self):
        """Register all builtin strategies."""
        strategies_to_register = [
            ('ConservativeMeanReversion', ConservativeMeanReversion),
            ('AggressiveMeanReversion', AggressiveMeanReversion),
            ('DivergenceCatcher', DivergenceCatcher),
            ('MultiTimeframeConfirmation', MultiTimeframeConfirmation),
            ('VolatilityFilterStrategy', VolatilityFilterStrategy),
        ]

        for strategy_id, strategy_class in strategies_to_register:
            try:
                instance = strategy_class()
                self._strategies[strategy_id] = strategy_class
                self._strategy_instances[strategy_id] = instance
                self._strategy_info[strategy_id] = instance.get_strategy_info()
            except Exception as e:
                print(f"Failed to register strategy {strategy_id}: {e}")

    def get_strategy(self, strategy_id: str) -> BaseStrategy:
        """Get strategy instance."""
        if strategy_id not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(f"Strategy '{strategy_id}' not found. Available: {available}")
        return self._strategy_instances[strategy_id]

    def get_strategy_info(self, strategy_id: str) -> Dict[str, Any]:
        """Get strategy information."""
        return self._strategy_info.get(strategy_id, {})

    def list_strategies(self) -> Dict[str, Dict[str, Any]]:
        """List all available strategies."""
        return self._strategy_info.copy()

    def recommend_strategies(self, user_profile: Dict[str, Any]) -> List[Tuple[str, int, List[str]]]:
        """Recommend strategies based on user profile."""
        experience_level = user_profile.get('experience_level', 'Beginner')
        risk_tolerance = user_profile.get('risk_tolerance', 'Medium')
        market_conditions = user_profile.get('market_conditions', 'Range-bound')

        recommendations = []

        for strategy_id, info in self._strategy_info.items():
            score = 50  # Base score
            reasons = []

            # Experience level matching
            complexity = info.get('complexity_level', 'Unknown')
            if experience_level == 'Beginner' and complexity == 'Beginner':
                score += 15
                reasons.append("Appropriate complexity for beginners")
            elif experience_level == 'Advanced' and complexity == 'Advanced':
                score += 10
                reasons.append("Matches advanced trader preferences")

            # Risk tolerance matching
            risk_level = info.get('risk_level', 'Medium')
            if risk_tolerance == risk_level:
                score += 20
                reasons.append(f"Matches {risk_tolerance.lower()} risk tolerance")
            elif (risk_tolerance == 'Low' and risk_level == 'Medium') or \
                 (risk_tolerance == 'Medium' and risk_level in ['Low', 'High']):
                score += 5
                reasons.append("Reasonable risk level match")

            # Market condition matching
            market_fit = info.get('market_fit', '').lower()
            if market_conditions.lower() in market_fit:
                score += 15
                reasons.append(f"Well-suited for {market_conditions.lower()} markets")

            recommendations.append((strategy_id, score, reasons))

        # Sort by score descending
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
