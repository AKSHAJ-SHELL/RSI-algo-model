"""
Backtest orchestrator - coordinates data fetching, indicator calculation, and backtesting.
"""

import pandas as pd
from datetime import date, timedelta
from typing import Dict, Any, Optional
import time

from src.data.fetcher import fetch_with_retry
from src.core.indicators import calculate_rsi, calculate_multiple_smas, calculate_zscore, detect_market_regime
from src.core.backtest import run_backtest_engine
from src.data.database import save_backtest_result
from src.utils.errors import DataFetchError, BacktestError


def run_backtest(
    ticker: str,
    start_date: date,
    end_date: date,
    config: Optional[Dict[str, Any]] = None,
    enable_out_of_sample: bool = False
) -> Dict[str, Any]:
    """
    Orchestrate complete backtest workflow.

    Args:
        ticker: Stock ticker symbol
        start_date: Backtest start date
        end_date: Backtest end date
        config: Strategy configuration (uses defaults if None)
        enable_out_of_sample: Enable out-of-sample testing with train/test split

    Returns:
        Dictionary with backtest results and metadata

    Raises:
        DataFetchError: If data fetching fails
        BacktestError: If backtest execution fails
    """
    if config is None:
        config = get_default_config()

    # Out-of-sample testing setup
    out_of_sample_results = {}
    if enable_out_of_sample:
        # Calculate train/test split (70/30)
        total_days = (end_date - start_date).days
        train_days = int(total_days * 0.7)
        split_date = start_date + timedelta(days=train_days)

        logger.info(f"Out-of-sample testing enabled: train={start_date} to {split_date}, "
                   f"test={split_date} to {end_date}")

        # Run training period backtest
        train_result = _run_single_backtest(ticker, start_date, split_date, config.copy())
        out_of_sample_results['training'] = train_result

        # Run test period backtest with same config (no re-optimization)
        test_config = config.copy()
        test_config['enable_statistical_validation'] = False  # Avoid duplicate validation
        test_result = _run_single_backtest(ticker, split_date, end_date, test_config)
        out_of_sample_results['testing'] = test_result

        # Analyze overfitting
        overfitting_analysis = _analyze_overfitting(train_result, test_result)
        out_of_sample_results['overfitting_analysis'] = overfitting_analysis

        logger.info(f"Out-of-sample analysis: overfitted={overfitting_analysis['overfitted']}, "
                   f"performance_degradation={overfitting_analysis['performance_degradation']:.2f}")

        # Return combined results
        return _format_out_of_sample_results(
            ticker, start_date, end_date, config,
            out_of_sample_results, overfitting_analysis
        )

    # Run single backtest (standard mode)
    return _run_single_backtest(ticker, start_date, end_date, config)


def _perform_statistical_validation(
    ticker: str,
    start_date: date,
    end_date: date,
    trades: pd.DataFrame,
    data: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform statistical validation of the strategy.

    Args:
        ticker: Strategy ticker
        start_date: Backtest start date
        end_date: Backtest end date
        trades: Trades DataFrame
        data: Price data DataFrame
        config: Strategy configuration

    Returns:
        Dictionary with validation results
    """
    validation_results = {}

    try:
        # Calculate strategy returns (daily equity curve)
        if trades.empty:
            return {'error': 'No trades to validate'}

        # Create daily returns series from trades
        strategy_returns = _calculate_strategy_returns(trades, data, start_date, end_date)

        if len(strategy_returns) < 30:  # Need minimum data for meaningful validation
            return {'error': 'Insufficient data for statistical validation'}

        # Fetch benchmark data (SPY)
        benchmark_data = None
        try:
            rate_limiter.wait_if_needed()
            benchmark_data = fetch_with_retry('SPY', start_date, end_date)
        except Exception as e:
            logger.warning(f"Failed to fetch SPY benchmark data: {e}")

        if benchmark_data is not None and not benchmark_data.empty:
            # Align benchmark returns with strategy returns
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()
            common_dates = strategy_returns.index.intersection(benchmark_returns.index)
            strategy_returns_aligned = strategy_returns.loc[common_dates]
            benchmark_returns_aligned = benchmark_returns.loc[common_dates]

            if len(strategy_returns_aligned) >= 30:
                # Test 1: Strategy vs Benchmark (SPY)
                edge_validation = validate_strategy_edge(
                    strategy_returns_aligned,
                    benchmark_returns_aligned,
                    confidence_level=0.95
                )
                validation_results['vs_benchmark'] = edge_validation

                # Test 2: Strategy vs Random Entry
                random_test = test_vs_random_entry(trades, data)
                validation_results['vs_random'] = random_test

                # Overall validation status
                has_edge = edge_validation.get('has_edge', False)
                beats_random = random_test.get('beats_random', False)

                validation_results['overall_validation'] = {
                    'passes_validation': has_edge and beats_random,
                    'has_edge_vs_market': has_edge,
                    'beats_random_entry': beats_random,
                    'confidence_level': 0.95
                }

                # Warnings for failed validation
                warnings = []
                if not has_edge:
                    warnings.append("Strategy does not show statistical edge vs SPY benchmark")
                if not beats_random:
                    warnings.append("Strategy does not beat random entry/exit")
                if warnings:
                    validation_results['warnings'] = warnings

                logger.info(f"Statistical validation for {ticker}: has_edge={has_edge}, "
                           f"beats_random={beats_random}")

        else:
            validation_results['error'] = 'Could not fetch benchmark data for validation'

    except Exception as e:
        logger.error(f"Statistical validation error for {ticker}: {e}")
        validation_results['error'] = str(e)

    return validation_results


def _run_single_backtest(
    ticker: str,
    start_date: date,
    end_date: date,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a single backtest period (helper for out-of-sample testing).

    This is the core backtest logic extracted for reuse.
    """
    start_time = time.time()

    try:
        # Step 1: Fetch data (with caching)
        cache = get_cache()
        cache_key = make_cache_key(ticker, start_date, end_date)
        data = cache.get(cache_key)

        if data is None:
            # Apply rate limiting before API calls
            rate_limiter.wait_if_needed()
            data = fetch_with_retry(ticker, start_date, end_date)
            if data is not None:
                cache.set(cache_key, data)

        if data is None or data.empty:
            raise DataFetchError(f"Failed to fetch data for {ticker}")

        logger.debug(f"Backtest {ticker}: Fetched {len(data)} data points from {start_date} to {end_date}")

        # Step 2: Detect market regime and adapt parameters
        try:
            regime = detect_market_regime(data, lookback=60)
            logger.info(f"Detected market regime for {ticker}: {regime.value}")

            # Adapt configuration based on regime
            config = get_regime_adaptive_parameters(config, regime)
        except Exception as e:
            logger.warning(f"Market regime detection failed for {ticker}, using base parameters: {e}")

        # RSI Safeguards: Disable signals in trending markets
        if config.get('disable_rsi_in_trends', True) and 'regime' in locals():
            from src.indicators.regime import MarketRegime
            if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                logger.warning(f"Backtest {ticker}: RSI mean-reversion DISABLED in {regime.value} market - "
                              "RSI works poorly in trends")

                # Return empty results for trending markets
                empty_trades = pd.DataFrame()
                empty_metrics = {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'total_trades': 0,
                    'warning': f'RSI signals disabled in {regime.value} market',
                    'regime': regime.value
                }

                return {
                    'ticker': ticker,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'config': config,
                    'data_points': len(data),
                    'data': data.to_dict('records') if not data.empty else [],
                    'trades': [],
                    'metrics': empty_metrics,
                    'statistical_validation': {},
                    'success': True,
                    'execution_time': time.time() - start_time,
                    'warning': f'RSI signals disabled in {regime.value} market'
                }

        # Step 3: Calculate indicators
        data = _add_indicators(data, config)
        logger.debug(f"Backtest {ticker}: Calculated indicators: {list(data.columns)}")

        # Step 3: Run backtest
        trades, metrics = run_backtest_engine(data, config)
        logger.debug(f"Backtest {ticker}: Generated {len(trades)} trades, metrics calculated")

        # Step 3.5: Statistical validation (optional)
        validation_results = {}
        if config.get('enable_statistical_validation', True) and not trades.empty:
            try:
                validation_results = _perform_statistical_validation(
                    ticker, start_date, end_date, trades, data, config
                )
                logger.debug(f"Backtest {ticker}: Statistical validation completed")
            except Exception as e:
                logger.warning(f"Statistical validation failed for {ticker}: {e}")
                validation_results = {'error': str(e)}

        # Step 4: Save results (optional)
        if config.get('save_results', True):
            try:
                save_backtest_result(ticker, start_date, end_date, config, metrics)
            except Exception as e:
                # Log but don't fail the backtest
                print(f"Warning: Failed to save results to database: {e}")

        # Step 5: Format results
        result = {
            'ticker': ticker,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'config': config,
            'data_points': len(data),
            'data': data.to_dict('records') if not data.empty else [],  # Include raw price data
            'trades': trades.to_dict('records') if not trades.empty else [],
            'metrics': metrics,
            'statistical_validation': validation_results,
            'success': True,
            'execution_time': time.time() - start_time,  # Would be set by caller
        }

        # Add validation warnings to main metrics for prominence
        if 'warnings' in validation_results:
            result['validation_warnings'] = validation_results['warnings']
            # Also add to metrics for UI display
            if 'validation_warnings' not in metrics:
                metrics['validation_warnings'] = validation_results['warnings']

        return result

    except Exception as e:
        raise BacktestError(f"Backtest failed for {ticker}: {str(e)}")


def _analyze_overfitting(train_result: Dict, test_result: Dict) -> Dict[str, Any]:
    """
    Analyze if strategy shows signs of overfitting.

    Args:
        train_result: Training period backtest results
        test_result: Testing period backtest results

    Returns:
        Dictionary with overfitting analysis
    """
    train_sharpe = train_result.get('metrics', {}).get('sharpe_ratio', 0)
    test_sharpe = test_result.get('metrics', {}).get('sharpe_ratio', 0)

    # Performance degradation indicates overfitting
    performance_degradation = train_sharpe - test_sharpe

    # Significant degradation (>0.5) suggests overfitting
    overfitted = performance_degradation > 0.5

    analysis = {
        'train_sharpe': float(train_sharpe),
        'test_sharpe': float(test_sharpe),
        'performance_degradation': float(performance_degradation),
        'overfitted': overfitted,
        'train_return': train_result.get('metrics', {}).get('total_return', 0),
        'test_return': test_result.get('metrics', {}).get('total_return', 0),
        'train_trades': len(train_result.get('trades', [])),
        'test_trades': len(test_result.get('trades', []))
    }

    return analysis


def _format_out_of_sample_results(
    ticker: str,
    start_date: date,
    end_date: date,
    config: Dict[str, Any],
    out_of_sample_results: Dict,
    overfitting_analysis: Dict
) -> Dict[str, Any]:
    """
    Format combined out-of-sample test results.

    Args:
        ticker: Ticker symbol
        start_date: Overall start date
        end_date: Overall end date
        config: Strategy configuration
        out_of_sample_results: Results from train/test periods
        overfitting_analysis: Overfitting analysis results

    Returns:
        Formatted combined results
    """
    # Use test period results as primary results
    primary_result = out_of_sample_results['testing'].copy()

    # Add out-of-sample metadata
    primary_result.update({
        'out_of_sample_testing': True,
        'training_period': out_of_sample_results['training'],
        'testing_period': out_of_sample_results['testing'],
        'overfitting_analysis': overfitting_analysis,
        'combined_analysis': {
            'train_vs_test_consistency': not overfitting_analysis['overfitted'],
            'recommendation': 'Consider different parameters' if overfitting_analysis['overfitted']
                            else 'Parameters appear robust'
        }
    })

    # Add overfitting warnings to main result
    if overfitting_analysis['overfitted']:
        warnings = primary_result.get('validation_warnings', [])
        warnings.append(f"Overfitting detected: {overfitting_analysis['performance_degradation']:.2f} "
                       "Sharpe degradation from train to test")
        primary_result['validation_warnings'] = warnings
        primary_result['metrics']['validation_warnings'] = warnings

    return primary_result


def _calculate_strategy_returns(
    trades: pd.DataFrame,
    data: pd.DataFrame,
    start_date: date,
    end_date: date
) -> pd.Series:
    """
    Calculate daily strategy returns from trades.

    Args:
        trades: Trades DataFrame
        data: Price data DataFrame
        start_date: Start date
        end_date: End date

    Returns:
        Daily returns series
    """
    # Create equity curve from trades
    equity_curve = [10000.0]  # Starting equity
    dates = [pd.Timestamp(start_date)]

    # Group trades by date
    if not trades.empty:
        daily_trades = trades.groupby(trades['date'].dt.date)

        current_equity = 10000.0
        current_date = start_date

        while current_date <= end_date:
            day_trades = daily_trades.get_group(current_date) if current_date in daily_trades.groups else pd.DataFrame()

            if not day_trades.empty:
                # Calculate daily P&L
                daily_pnl = day_trades['pnl'].sum() if 'pnl' in day_trades.columns else 0
                current_equity += daily_pnl

            equity_curve.append(current_equity)
            dates.append(pd.Timestamp(current_date))

            current_date = current_date + pd.Timedelta(days=1)

    # Convert to returns
    equity_series = pd.Series(equity_curve, index=dates)
    returns = equity_series.pct_change().dropna()

    return returns


def _add_indicators(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Add technical indicators to data.

    Args:
        data: OHLCV data
        config: Strategy configuration

    Returns:
        DataFrame with indicators added
    """
    # RSI
    rsi_period = config.get('rsi_period', 14)
    data['RSI_14'] = calculate_rsi(data['Close'], rsi_period)

    # Moving averages
    # Use shorter SMA periods if we don't have enough data for 200-period SMA
    data_length = len(data)
    if data_length >= 200:
        sma_periods = [20, 50, 200]
    elif data_length >= 50:
        sma_periods = [20, 50]
    else:
        sma_periods = [20] if data_length >= 20 else []

    smas = calculate_multiple_smas(data['Close'], sma_periods) if sma_periods else pd.DataFrame()
    data = pd.concat([data, smas], axis=1)

    # Z-score (optional)
    if config.get('enable_zscore', False):
        zscore_window = config.get('zscore_window', 50)
        data['zscore'] = calculate_zscore(data['Close'], zscore_window)

    return data


def run_multiple_backtests(
    tickers: list[str],
    start_date: date,
    end_date: date,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run backtests for multiple tickers.

    Args:
        tickers: List of ticker symbols
        start_date: Backtest start date
        end_date: Backtest end date
        config: Strategy configuration

    Returns:
        Dictionary with results for all tickers
    """
    results = {}
    successful = 0
    failed = 0

    for ticker in tickers:
        try:
            result = run_backtest(ticker, start_date, end_date, config)
            results[ticker] = result
            successful += 1
        except Exception as e:
            results[ticker] = {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }
            failed += 1

    return {
        'results': results,
        'summary': {
            'total_tickers': len(tickers),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(tickers) if tickers else 0
        }
    }


def get_default_config() -> Dict[str, Any]:
    """
    Get default strategy configuration.

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
        'min_volume': 0,

        # Risk management
        'position_size_pct': 2.0,  # 2% of capital per trade
        'slippage_pct': 0.001,     # 0.1% round-trip slippage
        'commission_per_share': 0.01,

        # Indicators
        'rsi_period': 14,
        'zscore_window': 50,

        # Other
        'save_results': True,
    }
