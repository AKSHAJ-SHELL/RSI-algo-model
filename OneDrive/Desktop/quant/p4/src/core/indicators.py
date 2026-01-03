"""
Consolidated technical indicators for RSI trading system.

Combines RSI, moving averages, swing detection, market regime analysis,
z-score, ATR, and divergence detection into a single module.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    VOLATILE = "volatile"
    CALM = "calm"


# RSI Indicators
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index) for a price series.

    Args:
        prices: Series of closing prices
        period: Lookback period (default 14)

    Returns:
        Series with RSI values (0-100 scale)
    """
    if period < 1:
        raise ValueError("RSI period must be >= 1")
    if len(prices) < period + 1:
        raise ValueError(f"Insufficient data for RSI calculation. Need at least {period + 1} prices, got {len(prices)}")

    # Calculate price changes
    delta = prices.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate initial averages
    avg_gain = gains.iloc[1:period+1].mean()
    avg_loss = losses.iloc[1:period+1].mean()

    # Initialize RSI series
    rsi_values = [None] * period

    # Calculate RSI for the rest of the series
    for i in range(period, len(prices)):
        if i == period:
            rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
            rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100.0
        else:
            gain = gains.iloc[i]
            loss = losses.iloc[i]
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

            rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
            rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100.0

        rsi_values.append(rsi)

    return pd.Series(rsi_values, index=prices.index, name=f'RSI_{period}')


def get_rsi_signal(rsi: pd.Series, oversold_threshold: float = 30, overbought_threshold: float = 70) -> pd.Series:
    """Generate RSI-based signals."""
    signals = pd.Series(0, index=rsi.index, name='rsi_signal')
    signals[rsi <= oversold_threshold] = 1    # Oversold - potential buy
    signals[rsi >= overbought_threshold] = -1  # Overbought - potential sell
    return signals


def calculate_dynamic_rsi_thresholds(prices: pd.Series, base_oversold: float = 30.0,
                                   base_overbought: float = 70.0, volatility_lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
    """Calculate volatility-adjusted RSI thresholds."""
    returns = prices.pct_change()
    volatility = returns.rolling(window=volatility_lookback).std()
    avg_volatility = volatility.rolling(window=volatility_lookback * 3).mean()
    avg_volatility = avg_volatility.replace(0, avg_volatility.mean())

    volatility_ratio = volatility / avg_volatility

    oversold_adjustment = pd.Series(0.0, index=prices.index)
    overbought_adjustment = pd.Series(0.0, index=prices.index)

    # High volatility: widen thresholds
    high_vol_mask = volatility_ratio > 1.5
    high_vol_ratio = (volatility_ratio[high_vol_mask] - 1.5) / 1.0
    high_vol_ratio = high_vol_ratio.clip(0, 1)
    oversold_adjustment[high_vol_mask] = -10.0 * high_vol_ratio
    overbought_adjustment[high_vol_mask] = 10.0 * high_vol_ratio

    # Low volatility: tighten thresholds
    low_vol_mask = volatility_ratio < 0.7
    low_vol_ratio = (0.7 - volatility_ratio[low_vol_mask]) / 0.3
    low_vol_ratio = low_vol_ratio.clip(0, 1)
    oversold_adjustment[low_vol_mask] = 5.0 * low_vol_ratio
    overbought_adjustment[low_vol_mask] = -5.0 * low_vol_ratio

    oversold_thresholds = base_oversold + oversold_adjustment
    overbought_thresholds = base_overbought + overbought_adjustment

    return oversold_thresholds.clip(15, 45), overbought_thresholds.clip(55, 85)


def get_dynamic_rsi_signals(rsi: pd.Series, prices: pd.Series, base_oversold: float = 30.0,
                          base_overbought: float = 70.0, volatility_lookback: int = 20) -> pd.Series:
    """Generate RSI signals using dynamic volatility-adjusted thresholds."""
    oversold_thresholds, overbought_thresholds = calculate_dynamic_rsi_thresholds(
        prices, base_oversold, base_overbought, volatility_lookback)

    signals = pd.Series(0, index=rsi.index, name='dynamic_rsi_signal')
    signals[rsi <= oversold_thresholds] = 1    # Oversold - potential buy
    signals[rsi >= overbought_thresholds] = -1  # Overbought - potential sell
    return signals


# Moving Average Indicators
def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    if period < 1:
        raise ValueError("SMA period must be >= 1")
    if len(prices) < period:
        raise ValueError(f"Insufficient data for SMA calculation. Need at least {period} prices, got {len(prices)}")
    return prices.rolling(window=period, min_periods=period).mean()


def calculate_multiple_smas(prices: pd.Series, periods: List[int]) -> pd.DataFrame:
    """Calculate multiple SMAs for different periods."""
    smas = {}
    for period in periods:
        sma = calculate_sma(prices, period)
        smas[f'SMA_{period}'] = sma
    return pd.DataFrame(smas)


def get_trend_signals(smas: pd.DataFrame) -> pd.Series:
    """Generate trend signals based on SMA relationships."""
    signals = pd.Series(0, index=smas.index, name='trend_signal')

    if 'SMA_20' not in smas.columns or 'SMA_200' not in smas.columns:
        return signals

    sma_20 = smas['SMA_20']
    sma_200 = smas['SMA_200']

    # Uptrend: SMA_20 > SMA_200
    uptrend_condition = (sma_20 > sma_200)
    signals[uptrend_condition] = 1

    # Downtrend: SMA_20 < SMA_200
    downtrend_condition = (sma_20 < sma_200)
    signals[downtrend_condition] = -1

    return signals


def get_sma_signals(prices: pd.Series, smas: pd.DataFrame) -> pd.Series:
    """Generate signals based on price vs SMA crossovers."""
    signals = pd.Series(0, index=prices.index, name='sma_signal')

    for col in smas.columns:
        if col.startswith('SMA_'):
            sma = smas[col]
            # Price crosses above SMA
            above_cross = (prices > sma) & (prices.shift(1) <= sma.shift(1))
            signals[above_cross] = 1

            # Price crosses below SMA
            below_cross = (prices < sma) & (prices.shift(1) >= sma.shift(1))
            signals[below_cross] = -1

    return signals


# Swing Point Detection
def detect_swing_highs_lows(data: pd.DataFrame, lookback: int = 5) -> Tuple[pd.Series, pd.Series]:
    """Detect swing highs and lows using vectorized operations."""
    if data.empty or len(data) < lookback * 2 + 1:
        return pd.Series(0, index=data.index, name='swing_high'), pd.Series(0, index=data.index, name='swing_low')

    highs = data['High']
    lows = data['Low']

    swing_highs = pd.Series(0, index=data.index, name='swing_high')
    swing_lows = pd.Series(0, index=data.index, name='swing_low')

    for i in range(lookback, len(data) - lookback):
        # Check swing high
        window_before_high = highs.iloc[i-lookback:i]
        window_after_high = highs.iloc[i+1:i+lookback+1]
        current_high = highs.iloc[i]

        if (current_high > window_before_high.max()) and (current_high > window_after_high.max()):
            swing_highs.iloc[i] = 1

        # Check swing low
        window_before_low = lows.iloc[i-lookback:i]
        window_after_low = lows.iloc[i+1:i+lookback+1]
        current_low = lows.iloc[i]

        if (current_low < window_before_low.min()) and (current_low < window_after_low.min()):
            swing_lows.iloc[i] = 1

    return swing_highs, swing_lows


def find_recent_swing_levels(data: pd.DataFrame, lookback: int = 5, max_levels: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Find the most recent swing high and low levels."""
    swing_highs, swing_lows = detect_swing_highs_lows(data, lookback)

    recent_highs = data.loc[swing_highs == 1, 'High'].tail(max_levels)
    resistance_levels = pd.Series(recent_highs.values, index=recent_highs.index, name='resistance')

    recent_lows = data.loc[swing_lows == 1, 'Low'].tail(max_levels)
    support_levels = pd.Series(recent_lows.values, index=recent_lows.index, name='support')

    return resistance_levels, support_levels


# Market Regime Detection
def detect_market_regime(data: pd.DataFrame, lookback: int = 60, volatility_threshold: float = 0.02,
                        trend_strength_threshold: float = 0.05) -> MarketRegime:
    """Detect the current market regime."""
    if data is None or data.empty or len(data) < lookback:
        return MarketRegime.RANGE_BOUND

    recent_data = data.tail(lookback)
    returns = recent_data['Close'].pct_change().dropna()
    volatility = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0

    start_price = recent_data['Close'].iloc[0]
    end_price = recent_data['Close'].iloc[-1]
    trend_strength = abs((end_price / start_price) - 1) if start_price > 0 else 0.0

    # Calculate trend consistency
    sma_20 = recent_data['Close'].rolling(20).mean()
    sma_50 = recent_data['Close'].rolling(50).mean()

    if len(sma_20.dropna()) > 10 and len(sma_50.dropna()) > 10:
        price_above_sma20 = (recent_data['Close'] > sma_20).tail(30).sum() / 30
        price_above_sma50 = (recent_data['Close'] > sma_50).tail(30).sum() / 30
        trend_consistency = max(price_above_sma20, 1 - price_above_sma20)
    else:
        trend_consistency = 0.5

    # Classify regime
    if volatility > volatility_threshold * 2:
        return MarketRegime.VOLATILE
    elif volatility < volatility_threshold * 0.5:
        return MarketRegime.CALM
    elif trend_strength > trend_strength_threshold and trend_consistency > 0.7:
        return MarketRegime.TRENDING_UP if end_price > start_price else MarketRegime.TRENDING_DOWN
    else:
        return MarketRegime.RANGE_BOUND


def get_regime_adaptive_parameters(base_config: Dict[str, Any], regime: MarketRegime) -> Dict[str, Any]:
    """Adjust strategy parameters based on detected market regime."""
    config = base_config.copy()

    if regime == MarketRegime.VOLATILE:
        config['oversold_threshold'] = min(35.0, config.get('oversold_threshold', 30.0) + 5)
        config['exit_threshold'] = max(65.0, config.get('exit_threshold', 50.0) - 5)
        config['position_size_pct'] = max(0.5, config.get('position_size_pct', 1.0) * 0.7)
        config['max_hold_days'] = min(30, config.get('max_hold_days', 60))

    elif regime == MarketRegime.CALM:
        config['oversold_threshold'] = max(25.0, config.get('oversold_threshold', 30.0) - 5)
        config['exit_threshold'] = min(75.0, config.get('exit_threshold', 50.0) + 5)
        config['position_size_pct'] = min(2.0, config.get('position_size_pct', 1.0) * 1.3)

    elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
        config['oversold_threshold'] = min(40.0, config.get('oversold_threshold', 30.0) + 10)
        config['exit_threshold'] = max(60.0, config.get('exit_threshold', 50.0) - 10)
        config['position_size_pct'] = max(0.3, config.get('position_size_pct', 1.0) * 0.5)
        config['require_uptrend'] = True

    return config


def get_regime_description(regime: MarketRegime) -> str:
    """Get human-readable description of market regime."""
    descriptions = {
        MarketRegime.TRENDING_UP: "Strong upward trend - RSI mean-reversion may fail",
        MarketRegime.TRENDING_DOWN: "Strong downward trend - RSI mean-reversion may fail",
        MarketRegime.RANGE_BOUND: "Sideways/range-bound - Ideal for RSI mean-reversion",
        MarketRegime.VOLATILE: "High volatility - Use wider thresholds and smaller positions",
        MarketRegime.CALM: "Low volatility - Can use tighter thresholds and larger positions"
    }
    return descriptions.get(regime, "Unknown regime")


# Z-Score Indicators
def calculate_zscore(prices: pd.Series, window: int = 50) -> pd.Series:
    """Calculate z-score: (price - SMA) / rolling_std."""
    if window < 2:
        raise ValueError("Z-score window must be >= 2")
    if len(prices) < window:
        raise ValueError(f"Insufficient data for z-score calculation. Need at least {window} prices, got {len(prices)}")

    rolling_mean = prices.rolling(window=window, min_periods=window).mean()
    rolling_std = prices.rolling(window=window, min_periods=window).std()
    zscore = (prices - rolling_mean) / rolling_std

    return zscore


def get_zscore_signals(zscore: pd.Series, oversold_threshold: float = -2.0, overbought_threshold: float = 2.0) -> pd.Series:
    """Generate z-score based signals."""
    signals = pd.Series(0, index=zscore.index, name='zscore_signal')
    signals[zscore <= oversold_threshold] = 1   # Oversold - potential buy
    signals[zscore >= overbought_threshold] = -1 # Overbought - potential sell
    return signals


# ATR (Average True Range)
def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, use_ema: bool = True) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close series must have the same length")
    if period < 1:
        raise ValueError("ATR period must be positive")

    if len(high) < period:
        return pd.Series([np.nan] * len(high), index=high.index, name=f'ATR_{period}')

    # Calculate True Range
    hl = high - low  # High - Low
    hc = (high - close.shift(1)).abs()  # |High - Previous Close|
    lc = (low - close.shift(1)).abs()   # |Low - Previous Close|
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # Calculate ATR
    if use_ema:
        # Use exponential moving average
        atr = tr.ewm(span=period, adjust=False).mean()
    else:
        # Use simple moving average
        atr = tr.rolling(window=period, min_periods=period).mean()

    return atr


# RSI Divergence Detection
def find_pivot_points(series: pd.Series, window: int = 5) -> Tuple[pd.Series, pd.Series]:
    """Find pivot highs and lows in a series."""
    pivot_highs = pd.Series(np.nan, index=series.index)
    pivot_lows = pd.Series(np.nan, index=series.index)

    for i in range(window, len(series) - window):
        # Check for pivot high
        if all(series.iloc[i] > series.iloc[i-j] for j in range(1, window+1)) and \
           all(series.iloc[i] > series.iloc[i+j] for j in range(1, window+1)):
            pivot_highs.iloc[i] = series.iloc[i]

        # Check for pivot low
        if all(series.iloc[i] < series.iloc[i-j] for j in range(1, window+1)) and \
           all(series.iloc[i] < series.iloc[i+j] for j in range(1, window+1)):
            pivot_lows.iloc[i] = series.iloc[i]

    return pivot_highs, pivot_lows


def detect_rsi_divergence(price_series: pd.Series, rsi_series: pd.Series,
                         min_divergence_periods: int = 10, max_divergence_periods: int = 50) -> pd.Series:
    """
    Detect RSI divergence patterns.

    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high
    """
    if len(price_series) != len(rsi_series):
        raise ValueError("Price and RSI series must have the same length")

    divergence_signals = pd.Series(0, index=price_series.index, name='rsi_divergence')

    # Find pivot points
    price_highs, price_lows = find_pivot_points(price_series)
    rsi_highs, rsi_lows = find_pivot_points(rsi_series)

    # Check for bullish divergence (lower price low, higher RSI low)
    for i in range(len(price_lows.dropna())):
        if i >= 2:  # Need at least 2 pivot lows for comparison
            current_price_low_idx = price_lows.dropna().index[i]
            current_rsi_low_idx = rsi_lows.dropna().index[i]

            # Find previous pivot lows within range
            prev_indices = [j for j in range(i-2, i) if j >= 0]

            for prev_i in prev_indices:
                prev_price_low_idx = price_lows.dropna().index[prev_i]
                prev_rsi_low_idx = rsi_lows.dropna().index[prev_i]

                # Check time separation
                periods_apart = abs(current_price_low_idx - prev_price_low_idx)
                if not (min_divergence_periods <= periods_apart <= max_divergence_periods):
                    continue

                # Bullish divergence: price made lower low, RSI made higher low
                price_lower_low = price_series.loc[current_price_low_idx] < price_series.loc[prev_price_low_idx]
                rsi_higher_low = rsi_series.loc[current_rsi_low_idx] > rsi_series.loc[prev_rsi_low_idx]

                if price_lower_low and rsi_higher_low:
                    divergence_signals.loc[current_price_low_idx] = 1  # Bullish divergence
                    break

    # Check for bearish divergence (higher price high, lower RSI high)
    for i in range(len(price_highs.dropna())):
        if i >= 2:  # Need at least 2 pivot highs for comparison
            current_price_high_idx = price_highs.dropna().index[i]
            current_rsi_high_idx = rsi_highs.dropna().index[i]

            # Find previous pivot highs within range
            prev_indices = [j for j in range(i-2, i) if j >= 0]

            for prev_i in prev_indices:
                prev_price_high_idx = price_highs.dropna().index[prev_i]
                prev_rsi_high_idx = rsi_highs.dropna().index[prev_i]

                # Check time separation
                periods_apart = abs(current_price_high_idx - prev_price_high_idx)
                if not (min_divergence_periods <= periods_apart <= max_divergence_periods):
                    continue

                # Bearish divergence: price made higher high, RSI made lower high
                price_higher_high = price_series.loc[current_price_high_idx] > price_series.loc[prev_price_high_idx]
                rsi_lower_high = rsi_series.loc[current_rsi_high_idx] < rsi_series.loc[prev_rsi_high_idx]

                if price_higher_high and rsi_lower_high:
                    divergence_signals.loc[current_price_high_idx] = -1  # Bearish divergence
                    break

    return divergence_signals
