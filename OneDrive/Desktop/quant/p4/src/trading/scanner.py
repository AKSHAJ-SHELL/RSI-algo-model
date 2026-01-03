"""
Live scanner for RSI trading signals.
"""

import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data.fetcher import fetch_with_retry
from src.core.indicators import detect_market_regime, MarketRegime, calculate_rsi, get_rsi_signal
from src.core.signals import generate_entry_signals, apply_entry_filters
from src.data.database import save_live_signals_batch, get_positions_batch
from src.data.validators import validate_ticker_list
from src.utils.logging import get_logger

logger = get_logger("scanner")


def scan_tickers(tickers: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Scan multiple tickers for RSI signals.

    Args:
        tickers: List of ticker symbols
        config: Scanner configuration

    Returns:
        List of signal dictionaries
    """
    if not tickers:
        return []

    # Validate tickers
    try:
        tickers = validate_ticker_list(tickers)
    except Exception as e:
        logger.error(f"Ticker validation failed: {e}")
        return []

    # Get current positions to avoid conflicting signals
    positions = get_positions_batch(tickers)

    signals = []
    max_workers = min(len(tickers), 5)  # Limit concurrent requests

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(scan_single_ticker, ticker, config, positions.get(ticker, 0)): ticker
            for ticker in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                signal = future.result()
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Failed to scan {ticker}: {e}")

    return signals


def scan_single_ticker(ticker: str, config: Dict[str, Any], current_position: int) -> Optional[Dict[str, Any]]:
    """
    Scan a single ticker for RSI signals.

    Args:
        ticker: Ticker symbol
        config: Scanner configuration
        current_position: Current position (0=none, 1=long)

    Returns:
        Signal dictionary or None
    """
    try:
        # Fetch data
        end_date = date.today()
        start_date = end_date - timedelta(days=365)  # 1 year for context

        data = fetch_with_retry(ticker, start_date, end_date)
        if data is None or data.empty:
            logger.warning(f"No data available for {ticker}")
            return None

        # RSI safeguards: Disable signals in trending markets
        if config.get('disable_rsi_in_trends', True):
            regime = detect_market_regime(data)
            if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                logger.info(f"Skipping {ticker}: RSI disabled in {regime.value} regime")
                return {
                    'ticker': ticker,
                    'signal': 'HOLD',
                    'reason': f'RSI disabled in {regime.value} market',
                    'regime': regime.value,
                    'rsi_value': None,
                    'price': data['Close'].iloc[-1] if not data.empty else None,
                    'timestamp': pd.Timestamp.now()
                }

        # Calculate indicators
        data['RSI_14'] = calculate_rsi(data['Close'])

        # Generate signals
        entry_signals = generate_entry_signals(data, config)
        entry_signals = apply_entry_filters(entry_signals, data, config)

        # Check for new entry signal
        latest_signal = entry_signals.iloc[-1] if not entry_signals.empty else 0

        signal_type = 'BUY' if latest_signal == 1 else 'HOLD'

        # Avoid conflicting signals
        if current_position > 0 and signal_type == 'BUY':
            signal_type = 'HOLD'
            reason = 'Already in position'
        else:
            reason = 'RSI oversold' if signal_type == 'BUY' else 'No signal'

        return {
            'ticker': ticker,
            'signal': signal_type,
            'reason': reason,
            'rsi_value': data['RSI_14'].iloc[-1] if 'RSI_14' in data.columns else None,
            'price': data['Close'].iloc[-1],
            'volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else None,
            'timestamp': pd.Timestamp.now()
        }

    except Exception as e:
        logger.error(f"Error scanning {ticker}: {e}")
        return None


def save_signals(signals: List[Dict[str, Any]]) -> None:
    """Save signals to database."""
    if not signals:
        return

    # Convert to database format
    db_signals = []
    for signal in signals:
        db_signals.append({
            'ticker': signal['ticker'],
            'signal': signal['signal'],
            'rsi_value': signal.get('rsi_value'),
            'price': signal.get('price'),
            'volume': signal.get('volume'),
            'created_at': signal.get('timestamp', pd.Timestamp.now()).isoformat()
        })

    try:
        save_live_signals_batch(db_signals)
        logger.info(f"Saved {len(db_signals)} signals to database")
    except Exception as e:
        logger.error(f"Failed to save signals: {e}")
