"""
Consolidated signal generation and filtering for RSI trading system.

Combines entry signals, exit signals, and filtering logic into a single module.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import date


# Entry Signals
def generate_entry_signals(data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """
    Generate entry signals based on RSI and optional filters.

    Args:
        data: DataFrame with OHLCV data and indicators
        config: Strategy configuration

    Returns:
        Series with entry signals: 1 (enter long), 0 (no signal)
    """
    signals = pd.Series(0, index=data.index, name='entry_signal')

    # Get RSI signals
    rsi_threshold = config.get('oversold_threshold', 30)
    if 'RSI_14' in data.columns:
        from .indicators import get_rsi_signal
        rsi_signals = get_rsi_signal(data['RSI_14'], oversold_threshold=rsi_threshold)
        signals[rsi_signals == 1] = 1

    # Apply z-score filter if enabled
    if config.get('enable_zscore_filter', False):
        zscore_threshold = config.get('zscore_oversold_threshold', -2.0)
        if 'zscore' in data.columns:
            from .indicators import get_zscore_signals
            zscore_signals = get_zscore_signals(
                data['zscore'],
                oversold_threshold=zscore_threshold
            )
            # Only keep RSI signals that also have z-score confirmation
            signals = signals & (zscore_signals == 1).astype(int)

    return signals


# Exit Signals
def generate_exit_signals(data: pd.DataFrame, entry_signals: pd.Series, config: Dict[str, Any]) -> pd.Series:
    """
    Generate exit signals based on RSI, trailing stops, and time limits.

    Args:
        data: DataFrame with OHLCV data and indicators
        entry_signals: Series with entry signals (to calculate holding periods)
        config: Strategy configuration

    Returns:
        Series with exit signals: 1 (exit position), 0 (hold)
    """
    signals = pd.Series(0, index=data.index)

    # RSI-based exit
    rsi_threshold = config.get('exit_threshold', 50)
    if 'RSI_14' in data.columns:
        overbought_mask = data['RSI_14'] >= rsi_threshold
        signals[overbought_mask] = 1

    # Time-based exit
    max_hold_days = config.get('max_hold_days', 42)  # 8 weeks default for swing trading
    min_hold_days = config.get('min_hold_days', 10)  # 2 weeks minimum for swing trading
    if max_hold_days > 0:
        time_signals = _generate_time_exits(entry_signals, min_hold_days, max_hold_days)
        signals = signals | time_signals

    # Trailing stop exit (if enabled)
    if config.get('enable_trailing_stop', False):
        trailing_signals = _generate_trailing_stop_exits(data, entry_signals, config)
        signals = signals | trailing_signals

    # Resistance-based profit taking (if enabled)
    if config.get('enable_resistance_exits', False):
        resistance_signals = _generate_resistance_exits(data, entry_signals, config)
        signals = signals | resistance_signals

    signals.name = 'exit_signal'
    return signals


def _generate_time_exits(entry_signals: pd.Series, min_hold_days: int, max_hold_days: int) -> pd.Series:
    """
    Generate time-based exit signals for swing trading.

    Swing trades should:
    - Hold minimum min_hold_days (e.g., 2 weeks)
    - Exit maximum max_hold_days (e.g., 8 weeks)
    """
    time_signals = pd.Series(0, index=entry_signals.index, name='time_exit')
    entry_dates = entry_signals[entry_signals == 1].index

    for entry_date in entry_dates:
        max_exit_date = entry_date + pd.Timedelta(days=max_hold_days)

        if max_exit_date in time_signals.index:
            time_signals.loc[max_exit_date] = 1
        elif max_exit_date > time_signals.index[-1]:
            time_signals.iloc[-1] = 1

    return time_signals


def _generate_trailing_stop_exits(data: pd.DataFrame, entry_signals: pd.Series, config: Dict[str, Any]) -> pd.Series:
    """Generate trailing stop exit signals."""
    trailing_signals = pd.Series(0, index=data.index, name='trailing_stop_exit')
    stop_pct = config.get('trailing_stop_pct', 5.0) / 100.0

    # Track highest price since entry
    in_position = entry_signals.cumsum().astype(bool)
    if not in_position.any():
        return trailing_signals

    # Calculate running maximum price since entry
    prices = data['High']
    running_max = prices.groupby(entry_signals.cumsum()).cummax()

    # Calculate trailing stop level
    trailing_stop = running_max * (1 - stop_pct)

    # Exit when price drops below trailing stop
    trailing_signals[data['Low'] <= trailing_stop] = 1
    trailing_signals = trailing_signals & in_position.astype(int)

    return trailing_signals


def _generate_resistance_exits(data: pd.DataFrame, entry_signals: pd.Series, config: Dict[str, Any]) -> pd.Series:
    """Generate profit-taking exit signals at resistance levels (swing highs)."""
    resistance_signals = pd.Series(0, index=data.index, name='resistance_exit')

    # Requires swing high detection
    if 'swing_high' not in data.columns:
        return resistance_signals

    # Only consider resistance exits if currently in a position
    in_position = entry_signals.cumsum().astype(bool)

    # Identify swing highs that occur while in a position
    potential_exits = data['swing_high'] == 1
    resistance_signals = potential_exits & in_position

    return resistance_signals.astype(int)


# Signal Filters
def apply_entry_filters(signals: pd.Series, data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """
    Apply additional filters to entry signals.

    Args:
        signals: Raw entry signals
        data: DataFrame with OHLCV data and indicators
        config: Strategy configuration

    Returns:
        Filtered entry signals
    """
    filtered_signals = signals.copy()

    # Trend continuation filter: require confirmed uptrend
    if config.get('require_trend_continuation', False):
        filtered_signals = apply_trend_continuation_filter(filtered_signals, data, config)
    elif config.get('require_uptrend', True):
        # Fallback to simple uptrend filter
        if 'SMA_200' in data.columns:
            uptrend_mask = data['Close'] > data['SMA_200']
            filtered_signals = filtered_signals & uptrend_mask.astype(int)

    # Liquidity filter (preferred over simple volume filter)
    if config.get('enable_liquidity_filter', False):
        min_avg_volume = config.get('min_avg_volume', 1000000)
        min_price = config.get('min_price', 5.0)
        filtered_signals = apply_liquidity_filter(filtered_signals, data, min_avg_volume, min_price)
    else:
        # Legacy minimum volume filter
        min_volume = config.get('min_volume', 0)
        if min_volume > 0:
            volume_mask = data['Volume'] >= min_volume
            filtered_signals = filtered_signals & volume_mask.astype(int)

    # Event filter (earnings, Fed meetings, etc.)
    if config.get('enable_event_filter', False):
        ticker = config.get('ticker', 'UNKNOWN')
        event_dates = config.get('event_dates', None)
        filtered_signals = apply_event_filter(
            filtered_signals, ticker, data, event_dates,
            days_before=config.get('event_filter_days_before', 3),
            days_after=config.get('event_filter_days_after', 3)
        )

    return filtered_signals


def apply_uptrend_filter(signals: pd.Series, data: pd.DataFrame, sma_period: int = 200) -> pd.Series:
    """Filter signals to only allow entries in uptrends."""
    sma_col = f'SMA_{sma_period}'

    if sma_col not in data.columns:
        return signals

    trend_mask = data['Close'] > data[sma_col]
    return signals & trend_mask.astype(int)


def apply_trend_continuation_filter(signals: pd.Series, data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """Filter signals to only allow entries in confirmed uptrends."""
    if not config.get('require_trend_continuation', False):
        return signals

    sma_20 = data.get('SMA_20')
    sma_50 = data.get('SMA_50')
    sma_200 = data.get('SMA_200')

    if sma_20 is None or sma_50 is None:
        return signals

    # Uptrend: price > SMA20 > SMA50 (and optionally SMA200)
    uptrend_mask = (data['Close'] > sma_20) & (sma_20 > sma_50)

    if sma_200 is not None:
        uptrend_mask = uptrend_mask & (sma_50 > sma_200)

    # Additional check: SMAs should be sloping up
    if len(sma_20) > 5:
        sma_20_slope = sma_20 > sma_20.shift(5)
        uptrend_mask = uptrend_mask & sma_20_slope

    return signals & uptrend_mask.astype(int)


def apply_volume_filter(signals: pd.Series, data: pd.DataFrame, min_volume: int = 1000000) -> pd.Series:
    """Filter signals based on minimum volume."""
    volume_mask = data['Volume'] >= min_volume
    return signals & volume_mask.astype(int)


def apply_gap_filter(signals: pd.Series, data: pd.DataFrame, max_gap_pct: float = 5.0) -> pd.Series:
    """Filter signals to avoid gap-down entries."""
    prev_close = data['Close'].shift(1)
    gap_pct = ((data['Open'] - prev_close) / prev_close) * 100
    gap_mask = gap_pct >= -max_gap_pct
    return signals & gap_mask.astype(int)


def apply_earnings_filter(signals: pd.Series, data: pd.DataFrame, earnings_dates: List[str] = None) -> pd.Series:
    """Filter signals to avoid entries near earnings dates."""
    if not earnings_dates:
        return signals

    earnings_dt = pd.to_datetime(earnings_dates)
    mask = pd.Series(True, index=data.index)

    for earnings_date in earnings_dt:
        start_date = earnings_date - pd.Timedelta(days=2)
        end_date = earnings_date + pd.Timedelta(days=2)
        date_range = (data.index >= start_date) & (data.index <= end_date)
        mask &= ~date_range

    return signals & mask.astype(int)


def apply_liquidity_filter(signals: pd.Series, data: pd.DataFrame, min_avg_volume: int = 1000000,
                          min_price: float = 5.0) -> pd.Series:
    """Filter signals based on liquidity requirements."""
    # Price filter
    price_mask = data['Close'] >= min_price

    # Volume filter (average volume over last 20 days)
    if len(data) >= 20:
        avg_volume = data['Volume'].rolling(20).mean()
        volume_mask = avg_volume >= min_avg_volume
    else:
        volume_mask = data['Volume'] >= min_avg_volume

    combined_mask = price_mask & volume_mask
    return signals & combined_mask.astype(int)


def apply_event_filter(signals: pd.Series, ticker: str, data: pd.DataFrame, event_dates: Optional[List[str]] = None,
                      days_before: int = 3, days_after: int = 3) -> pd.Series:
    """Filter signals to avoid entries near high-impact events."""
    if not event_dates:
        return signals

    mask = pd.Series(True, index=data.index)

    for event_date_str in event_dates:
        try:
            event_date = pd.to_datetime(event_date_str).date()
            start_date = event_date - pd.Timedelta(days=days_before)
            end_date = event_date + pd.Timedelta(days=days_after)

            date_range = (data.index.date >= start_date) & (data.index.date <= end_date)
            mask &= ~date_range
        except (ValueError, TypeError):
            continue  # Skip invalid dates

    return signals & mask.astype(int)


def combine_filters(signals: pd.Series, data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """Apply all configured filters to signals."""
    filtered_signals = signals.copy()

    if config.get('require_uptrend', True):
        filtered_signals = apply_uptrend_filter(filtered_signals, data, config.get('trend_sma_period', 200))

    min_volume = config.get('min_volume', 0)
    if min_volume > 0:
        filtered_signals = apply_volume_filter(filtered_signals, data, min_volume)

    max_gap_pct = config.get('max_gap_pct', 0)
    if max_gap_pct > 0:
        filtered_signals = apply_gap_filter(filtered_signals, data, max_gap_pct)

    earnings_dates = config.get('earnings_dates')
    if earnings_dates:
        filtered_signals = apply_earnings_filter(filtered_signals, data, earnings_dates)

    return filtered_signals


# Signal Details
def get_entry_signal_details(signals: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
    """Get detailed information about entry signals."""
    signal_dates = signals[signals == 1].index

    if len(signal_dates) == 0:
        return pd.DataFrame()

    details = []
    for date in signal_dates:
        detail = {
            'date': date,
            'price': data.loc[date, 'Close'],
            'rsi': data.loc[date, 'RSI_14'] if 'RSI_14' in data.columns else None,
            'zscore': data.loc[date, 'zscore'] if 'zscore' in data.columns else None,
            'volume': data.loc[date, 'Volume'],
            'sma_20': data.loc[date, 'SMA_20'] if 'SMA_20' in data.columns else None,
            'sma_50': data.loc[date, 'SMA_50'] if 'SMA_50' in data.columns else None,
            'sma_200': data.loc[date, 'SMA_200'] if 'SMA_200' in data.columns else None,
        }
        details.append(detail)

    return pd.DataFrame(details)
