"""
Live scanner orchestrator - coordinates scanning for current signals.
"""

import pandas as pd
from datetime import date, timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

from src.data.fetcher import fetch_with_retry
from src.core.indicators import calculate_rsi, detect_market_regime, MarketRegime
from src.core.signals import generate_entry_signals, apply_entry_filters, generate_exit_signals
from src.data.database import save_live_signals_batch, get_db_connection
from src.data.validators import validate_parameter_conflicts, validate_data_continuity
from src.utils.errors import DataFetchError
from src.logging.logger import get_logger

logger = get_logger("scanner_orchestrator")


def validate_data_quality(data: pd.DataFrame, ticker: str) -> Tuple[bool, Optional[str]]:
    """
    Validate data quality before signal generation.

    Comprehensive checks for missing data, gaps, suspicious movements, etc.

    Args:
        data: DataFrame with OHLCV data
        ticker: Ticker symbol for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if data.empty:
        return False, f"{ticker}: Data is empty"

    # Check for missing data
    missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
    if missing_pct > 0.1:  # More than 10% missing
        return False, f"{ticker}: >10% missing data ({missing_pct:.2%})"

    # Check for data gaps
    is_valid, gap_error = validate_data_continuity(data, max_gap_days=5)
    if not is_valid:
        return False, f"{ticker}: {gap_error}"

    # Check for suspicious price movements (possible data errors)
    if len(data) > 1:
        returns = data['Close'].pct_change()
        extreme_moves = (returns.abs() > 0.5).sum()  # >50% daily moves
        if extreme_moves > len(data) * 0.01:  # More than 1% of days
            return False, f"{ticker}: >1% extreme price movements ({extreme_moves} moves >50%) - possible data errors"

    # Check for zero or negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in data.columns and (data[col] <= 0).any():
            return False, f"{ticker}: Zero or negative prices in column '{col}'"

    # Check for reasonable volume (not all zeros)
    if 'Volume' in data.columns:
        zero_volume_days = (data['Volume'] == 0).sum()
        if zero_volume_days > len(data) * 0.5:  # More than 50% zero volume
            return False, f"{ticker}: >50% zero volume days ({zero_volume_days}) - possible delisted stock"

    return True, None


def validate_strategy_compatibility(strategy_id: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that strategy requirements are met.

    Args:
        strategy_id: Strategy identifier
        config: Strategy configuration

    Returns:
        Tuple of (is_compatible, missing_requirements)
    """
    from src.strategies.registry import get_strategy

    try:
        strategy = get_strategy(strategy_id)
        required_indicators = strategy.get_required_indicators()

        # Check if required indicators are in the config's calculated_indicators list
        calculated_indicators = config.get('calculated_indicators', [])
        missing = []

        for indicator in required_indicators:
            if indicator not in calculated_indicators:
                missing.append(indicator)

        return len(missing) == 0, missing

    except Exception as e:
        return False, [f"Strategy validation failed: {e}"]


def get_positions_batch(tickers: List[str], max_position_age_days: int = 90) -> Dict[str, int]:
    """
    Get current positions for multiple tickers in a single database query.

    Args:
        tickers: List of ticker symbols
        max_position_age_days: Maximum age for positions in days

    Returns:
        Dictionary mapping ticker to position status (0 = no position, 1 = long position)
    """
    positions = {}
    try:
        with get_db_connection() as conn:
            # Create placeholders for SQL query
            placeholders = ','.join(['?'] * len(tickers))
            cutoff_date = datetime.now() - timedelta(days=max_position_age_days)

            # Single query to get all recent positions
            cursor = conn.execute(f"""
                SELECT ticker, signal, created_at FROM live_signals
                WHERE ticker IN ({placeholders})
                AND created_at > ?
                ORDER BY ticker, created_at DESC
            """, (*tickers, cutoff_date.isoformat()))

            # Process results - take the most recent signal for each ticker
            for row in cursor.fetchall():
                ticker = row['ticker']
                if ticker not in positions:
                    # Check if position is still valid (not expired)
                    signal_age_days = (datetime.now() - datetime.fromisoformat(row['created_at'])).days
                    if signal_age_days <= max_position_age_days:
                        positions[ticker] = 1 if row['signal'] == 'BUY' else 0
                    else:
                        positions[ticker] = 0  # Expired position

    except Exception as e:
        logger.warning(f"Batch position check failed: {e}")

    # Fill in missing tickers with no position
    return {ticker: positions.get(ticker, 0) for ticker in tickers}


def scan_tickers(
    tickers: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Scan tickers for current trading signals.

    Args:
        tickers: List of ticker symbols to scan
        config: Strategy configuration

    Returns:
        Dictionary with scan results
    """
    if config is None:
        config = get_default_config()

    # Validate parameter conflicts before scanning
    param_errors = validate_parameter_conflicts(config)
    if param_errors:
        logger.error(f"Parameter conflicts detected: {param_errors}")
        return {
            'results': {},
            'summary': {
                'total_tickers': len(tickers),
                'successful_scans': 0,
                'failed_scans': len(tickers),
                'scan_timestamp': datetime.now().isoformat(),
                'error': f"Parameter conflicts: {param_errors}",
                'signals_by_type': {'ERROR': len(tickers)}
            }
        }

    results = {}
    successful = 0
    failed = 0

    # Calculate date range for analysis (last 3 months + some buffer)
    end_date = date.today()
    start_date = date(end_date.year - 1, end_date.month, end_date.day)

    # Batch position check for all tickers (single database query)
    logger.debug(f"Performing batch position check for {len(tickers)} tickers")
    positions = get_positions_batch(tickers, max_position_age_days=config.get('max_position_age_days', 90))

    # Use concurrent execution for better performance
    max_workers = min(10, len(tickers))  # Limit to 10 concurrent workers to be respectful
    signals_to_save = []  # Collect signals for batch saving

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scan tasks with position info
        future_to_ticker = {
            executor.submit(scan_single_ticker, ticker, start_date, end_date, config, positions.get(ticker, 0)): ticker
            for ticker in tickers
        }

        # Process completed tasks as they finish with timeout and isolation
        for future in as_completed(future_to_ticker, timeout=60):  # 60 second timeout per batch
            ticker = future_to_ticker[future]
            try:
                # Add timeout per individual ticker (30 seconds max)
                signal_data = future.result(timeout=30)
                results[ticker] = signal_data
                if signal_data.get('success', False):
                    successful += 1

                    # Collect signals for batch saving (only non-HOLD signals)
                    if signal_data['signal'] != 'HOLD':
                        signals_to_save.append({
                            'ticker': ticker,
                            'signal_date': end_date,
                            'current_rsi': signal_data.get('rsi'),
                            'zscore': signal_data.get('zscore'),
                            'sma_20': signal_data.get('sma_20'),
                            'sma_50': signal_data.get('sma_50'),
                            'sma_200': signal_data.get('sma_200'),
                            'signal': signal_data['signal'],
                            'entry_price': signal_data.get('price')
                        })
                else:
                    failed += 1

            except TimeoutError:
                logger.error(f"Ticker {ticker} timed out after 30 seconds")
                results[ticker] = {
                    'success': False,
                    'error': 'Timeout after 30 seconds',
                    'ticker': ticker,
                    'signal': 'ERROR'
                }
                failed += 1
            except Exception as e:
                logger.error(f"Ticker {ticker} failed with error: {e}", exc_info=True)
                results[ticker] = {
                    'success': False,
                    'error': str(e),
                    'ticker': ticker,
                    'signal': 'ERROR'
                }
                failed += 1

    # Batch save all signals at once
    if signals_to_save:
        try:
            save_live_signals_batch(signals_to_save)
            logger.info(f"Batch saved {len(signals_to_save)} signals to database")
        except Exception as e:
            logger.error(f"Failed to batch save signals: {e}")

    return {
        'results': results,
        'summary': {
            'total_tickers': len(tickers),
            'successful': successful,
            'failed': failed,
            'scan_timestamp': datetime.now().isoformat(),
            'config_summary': {
                'oversold_threshold': config.get('oversold_threshold'),
                'require_uptrend': config.get('require_uptrend'),
                'enable_zscore_filter': config.get('enable_zscore_filter'),
            }
        }
    }


def scan_single_ticker(
    ticker: str,
    start_date: date,
    end_date: date,
    config: Dict[str, Any],
    current_position: int = 0
) -> Dict[str, Any]:
    """
    Scan a single ticker for signals.

    Args:
        ticker: Ticker symbol
        start_date: Start date for data (may be extended for indicator calculation)
        end_date: End date for data
        config: Strategy configuration

    Returns:
        Dictionary with signal analysis
    """
    # Validate strategy compatibility before scanning
    strategy_id = config.get('strategy_id', 'conservative')
    is_compatible, missing_indicators = validate_strategy_compatibility(strategy_id, config)
    if not is_compatible:
        logger.error(f"Strategy {strategy_id} incompatible for {ticker}: missing {missing_indicators}")
        return {
            'ticker': ticker,
            'success': False,
            'error': f"Strategy requires indicators: {missing_indicators}",
            'signal': 'ERROR'
        }

    # Calculate exact date range needed for swing trading
    # Need enough data for longest indicators (200-day SMA) + swing lookback (6 months)
    required_days = 200 + 180  # 200 for SMA + 180 for 6-month swing analysis
    actual_start_date = end_date - timedelta(days=required_days)

    # Don't go before the provided start_date
    fetch_start_date = max(actual_start_date, start_date)

    logger.debug(f"Scanning {ticker}: start_date={fetch_start_date}, end_date={end_date}, required_days={required_days}")

    # Fetch data with optimized date range
    cache = get_cache()
    cache_key = make_cache_key(ticker, fetch_start_date, end_date)
    data = cache.get(cache_key)

    if data is None:
        # Apply rate limiting before API calls
        rate_limiter.wait_if_needed()
        data = fetch_with_retry(ticker, fetch_start_date, end_date)
        if data is not None:
            cache.set(cache_key, data)

    if data is None or data.empty:
        raise DataFetchError(f"No data available for {ticker}")

    logger.debug(f"{ticker}: Fetched {len(data)} data points")

    # Comprehensive data quality validation before signal generation
    is_valid, quality_error = validate_data_quality(data, ticker)
    if not is_valid:
        logger.warning(f"Data quality check failed for {ticker}: {quality_error}")
        return {
            'ticker': ticker,
            'success': False,
            'error': quality_error,
            'signal': 'ERROR'
        }

    # Validate minimum data requirements for swing trading
    min_days_required = max(200, config.get('rsi_period', 14) + 50)  # At least 200 days or RSI period + buffer
    if len(data) < min_days_required:
        return {
            'success': False,
            'error': f'Insufficient data: {len(data)} days available, need at least {min_days_required} days for reliable signals'
        }

    # Detect market regime and adapt parameters (before indicators)
    try:
        regime = detect_market_regime(data, lookback=60)
        logger.info(f"{ticker}: Detected market regime: {regime.value}")

        # Adapt strategy parameters based on regime
        config = get_regime_adaptive_parameters(config, regime)

        # Log regime-specific warnings
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            logger.warning(f"{ticker}: Trending market detected - RSI mean-reversion may underperform")
        elif regime == MarketRegime.RANGE_BOUND:
            logger.info(f"{ticker}: Range-bound market - optimal for RSI mean-reversion")
    except Exception as e:
        logger.warning(f"Market regime detection failed for {ticker}, using base parameters: {e}")
        regime = None  # Set to None if detection failed

    # RSI Safeguards: Disable signals in trending markets
    if config.get('disable_rsi_in_trends', True) and regime is not None:
        from src.indicators.regime import MarketRegime
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            logger.warning(f"{ticker}: RSI mean-reversion DISABLED in {regime.value} market - "
                          "RSI works poorly in trends")
            return {
                'ticker': ticker,
                'success': True,  # Not an error, just no signal
                'signal': 'HOLD',
                'reason': f'RSI disabled in {regime.value} market',
                'regime': regime.value,
                'warning': 'RSI mean-reversion signals disabled in trending markets'
            }

    # Calculate indicators
    data = _add_scanner_indicators(data, config)
    logger.debug(f"{ticker}: Calculated indicators: {list(data.columns)}")

    # Get current values
    current_price = data['Close'].iloc[-1]
    current_rsi = data['RSI_14'].iloc[-1] if 'RSI_14' in data.columns else None
    current_zscore = data['zscore'].iloc[-1] if 'zscore' in data.columns else None

    # Get SMAs
    sma_20 = data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns else None
    sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else None
    sma_200 = data['SMA_200'].iloc[-1] if 'SMA_200' in data.columns else None

    # Generate signals using 3-6 months of data for swing trading context
    min_days_for_swing = 90  # 3 months minimum
    lookback_days = min(180, len(data))  # Up to 6 months
    recent_data = data.tail(lookback_days) if len(data) >= min_days_for_swing else data

    # Current position state passed from batch check to prevent conflicting signals

    # Only generate entry signals if we don't have a position
    if current_position == 0:
        entry_signals = generate_entry_signals(recent_data, config)
        entry_signals = apply_entry_filters(entry_signals, recent_data, config)
        exit_signals = pd.Series(0, index=recent_data.index, name='exit_signal')  # No exit signals
        logger.debug(f"{ticker}: Generated {entry_signals.sum()} entry signals, {exit_signals.sum()} exit signals (no position)")
    else:
        # Only generate exit signals if we have a position
        entry_signals = pd.Series(0, index=recent_data.index, name='entry_signal')  # No entry signals
        exit_signals = generate_exit_signals(recent_data, entry_signals, config)
        logger.debug(f"{ticker}: Generated {entry_signals.sum()} entry signals, {exit_signals.sum()} exit signals (has position)")

    # Determine current signal
    current_entry = entry_signals.iloc[-1] if not entry_signals.empty else 0
    current_exit = exit_signals.iloc[-1] if not exit_signals.empty else 0

    if current_entry == 1:
        signal = 'BUY'
    elif current_exit == 1:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    # Get recent signals (last 30 days)
    recent_signals = get_live_signals(ticker)
    last_entry_date = None
    holding_days = 0

    if recent_signals:
        # Find most recent BUY signal
        buy_signals = [s for s in recent_signals if s['signal'] == 'BUY']
        if buy_signals:
            last_entry_date = buy_signals[-1]['signal_date']
            holding_days = (datetime.now().date() - date.fromisoformat(last_entry_date)).days

    return {
        'success': True,
        'ticker': ticker,
        'signal': signal,
        'price': current_price,
        'rsi': current_rsi,
        'zscore': current_zscore,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'sma_200': sma_200,
        'volume': data['Volume'].iloc[-1],
        'last_entry_date': last_entry_date,
        'holding_days': holding_days,
        'data_points': len(data),
        'scan_date': end_date.isoformat(),
    }


def _add_scanner_indicators(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Add indicators needed for scanning.

    Args:
        data: OHLCV data
        config: Strategy configuration

    Returns:
        DataFrame with indicators
    """
    # RSI
    rsi_period = config.get('rsi_period', 14)
    data['RSI_14'] = calculate_rsi(data['Close'], rsi_period)

    # Moving averages - adapt based on available data
    data_length = len(data)
    all_periods = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    sma_periods = [period for period in all_periods if period <= data_length]

    smas = calculate_multiple_smas(data['Close'], sma_periods) if sma_periods else pd.DataFrame()
    if not smas.empty:
        data = pd.concat([data, smas], axis=1)

    # Z-score (if enabled)
    if config.get('enable_zscore_filter', False):
        zscore_window = config.get('zscore_window', 50)
        data['zscore'] = calculate_zscore(data['Close'], zscore_window)

    return data


def get_current_position(ticker: str, max_position_age_days: int = 90) -> int:
    """
    Get current position for a ticker with time decay.

    Positions expire after max_position_age_days to handle stale signals.

    Args:
        ticker: Ticker symbol
        max_position_age_days: Maximum age for positions in days (default 90)

    Returns:
        0 if no position or expired, 1 if active long position
    """
    try:
        # Check database for most recent signal with timestamp
        with get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT signal, created_at FROM live_signals
                WHERE ticker = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (ticker,))

            row = cursor.fetchone()
            if row and row['signal'] == 'BUY':
                # Check if position has expired
                created_at = datetime.fromisoformat(row['created_at'])
                age_days = (datetime.now() - created_at).days

                # Position expires after max age
                if age_days > max_position_age_days:
                    logger.info(f"Position for {ticker} expired ({age_days} days old)")
                    return 0  # Position expired

                return 1  # We have an active position
            else:
                return 0  # No position or last signal was SELL
    except Exception as e:
        logger.warning(f"Could not check position for {ticker}: {e}")
        return 0  # Assume no position on error


def get_scanner_summary(signals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics for scan results.

    Args:
        signals: Scan results dictionary

    Returns:
        Summary statistics
    """
    results = signals.get('results', {})
    buy_signals = 0
    sell_signals = 0
    hold_signals = 0
    errors = 0

    tickers_by_signal = {
        'BUY': [],
        'SELL': [],
        'HOLD': [],
        'ERROR': []
    }

    for ticker, result in results.items():
        signal = result.get('signal', 'ERROR')
        tickers_by_signal[signal].append(ticker)

        if signal == 'BUY':
            buy_signals += 1
        elif signal == 'SELL':
            sell_signals += 1
        elif signal == 'HOLD':
            hold_signals += 1
        else:
            errors += 1

    return {
        'total_tickers': len(results),
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'hold_signals': hold_signals,
        'errors': errors,
        'tickers_by_signal': tickers_by_signal,
        'scan_timestamp': signals.get('summary', {}).get('scan_timestamp'),
    }


def get_default_config() -> Dict[str, Any]:
    """
    Get default scanner configuration.

    Returns:
        Dictionary with default configuration
    """
    return {
        'oversold_threshold': 30,
        'exit_threshold': 50,
        'require_uptrend': True,
        'enable_zscore_filter': False,
        'zscore_oversold_threshold': -2.0,
        'rsi_period': 14,
        'zscore_window': 50,
        'min_volume': 1000000,  # 1M shares minimum
    }
