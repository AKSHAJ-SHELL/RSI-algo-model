"""
Consolidated backtest engine for RSI trading strategies.

Combines position tracking, trade execution, and performance metrics calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime


def calculate_risk_adjusted_position_size(
    base_size_pct: float,
    volatility: float,
    target_volatility: float = 0.15,
    max_position_size: float = 5.0,
    min_position_size: float = 0.5
) -> float:
    """
    Calculate position size adjusted for stock volatility.

    Higher volatility = smaller position size to maintain constant risk.
    """
    if volatility <= 0:
        return base_size_pct

    volatility_ratio = target_volatility / volatility
    adjusted_size = base_size_pct * volatility_ratio

    return max(min_position_size, min(max_position_size, adjusted_size))


def _adjust_position_size_for_scaling(
    base_size: float,
    entry_price: float,
    current_price: float,
    scaling_levels: Optional[List[float]] = None
) -> float:
    """Adjust position size based on price action for scaling in."""
    if scaling_levels is None:
        scaling_levels = [0.95, 0.90, 0.85]  # Scale in at 5%, 10%, 15% down

    if entry_price <= 0:
        return base_size

    price_change_pct = (current_price / entry_price) - 1

    # Scale in on pullbacks (negative price change)
    if price_change_pct < 0:
        for level in sorted(scaling_levels, reverse=True):
            if price_change_pct <= (level - 1.0):
                scale_factor = 1.0 + abs(level - 1.0) * 2
                return base_size * scale_factor

    return base_size


def run_backtest_engine(
    data: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run vectorized backtest simulation with risk management.

    Args:
        data: DataFrame with OHLCV data and indicators
        config: Strategy configuration

    Returns:
        Tuple of (trades_df, metrics_dict)
    """
    # Apply risk-adjusted position sizing if enabled
    if config.get('use_risk_adjusted_sizing', False):
        returns = data['Close'].pct_change().dropna()
        if len(returns) > 0:
            stock_volatility = returns.std() * np.sqrt(252)

            base_size = config.get('position_size_pct', 1.0)
            adjusted_size = calculate_risk_adjusted_position_size(
                base_size, stock_volatility,
                target_volatility=config.get('target_portfolio_volatility', 0.15),
                max_position_size=config.get('max_position_size_pct', 5.0),
                min_position_size=config.get('min_position_size_pct', 0.5)
            )
            config['position_size_pct'] = adjusted_size

    # Generate entry and exit signals
    from .signals import generate_entry_signals, generate_exit_signals, apply_entry_filters

    entry_signals = generate_entry_signals(data, config)
    entry_signals = apply_entry_filters(entry_signals, data, config)
    exit_signals = generate_exit_signals(data, entry_signals, config)

    # Track positions and execute trades
    positions, trades, risk_metrics = _track_positions_with_risk_management(
        data, entry_signals, exit_signals, config
    )

    # Calculate metrics
    metrics = _calculate_performance_metrics(data, positions, trades, config)

    # Add risk metrics to results
    metrics.update(risk_metrics)

    return trades, metrics


def _track_positions_with_risk_management(
    data: pd.DataFrame,
    entry_signals: pd.Series,
    exit_signals: pd.Series,
    config: Dict[str, Any],
    risk_manager=None
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    """
    Track positions and execute trades with risk management integration.
    """
    # Shift signals to next day for execution (realistic timing)
    entry_exec_signals = entry_signals.shift(1).fillna(0).astype(int)
    exit_exec_signals = exit_signals.shift(1).fillna(0).astype(int)

    # Get execution prices (next day's open)
    entry_exec_prices = data['Open'].shift(-1)
    exit_exec_prices = data['Open'].shift(-1)

    # Initialize tracking variables
    base_position_size = config.get('position_size_pct', 1.0) / 100.0
    current_equity = 10000.0
    position_size = 0.0
    entry_price = 0.0

    # Tracking for risk management
    risk_metrics = {
        'trading_halted': False,
        'position_reductions': 0,
        'max_drawdown_experienced': 0.0,
        'trading_halt_date': None
    }

    # Initialize results
    positions = pd.Series(0.0, index=data.index)
    trades = []

    # Process each day sequentially for risk management
    for i, date in enumerate(data.index):
        # Update current equity (from previous day's position)
        if position_size != 0 and i > 0:
            prev_date = data.index[i-1]
            current_price = data.loc[date, 'Close']
            prev_price = data.loc[prev_date, 'Close']
            equity_change = position_size * current_equity * (current_price - prev_price) / prev_price
            current_equity += equity_change

        # Update risk manager with current equity
        if risk_manager is not None:
            risk_manager.update_equity(current_equity)

            # Check if trading should be halted
            if risk_manager.should_halt_trading():
                risk_metrics['trading_halted'] = True
                risk_metrics['trading_halt_date'] = date
                break

            # Get position multiplier based on drawdown
            position_multiplier = risk_manager.get_position_multiplier()

        else:
            position_multiplier = 1.0

        # Check for entry signal
        if entry_exec_signals.loc[date] == 1 and entry_exec_prices.loc[date] > 0:
            # Apply risk-adjusted position size
            adjusted_position_size = base_position_size * position_multiplier

            # Only enter if we have sufficient capital
            max_position_value = current_equity * adjusted_position_size
            if max_position_value > 0:
                entry_price = _calculate_entry_price(entry_exec_prices.loc[date], config)
                position_size = adjusted_position_size

                # Calculate actual shares
                max_shares = max_position_value // entry_price
                if max_shares > 0:
                    actual_position_value = max_shares * entry_price
                    position_size = actual_position_value / current_equity

                    trades.append({
                        'date': date,
                        'type': 'entry',
                        'price': entry_price,
                        'quantity': position_size,
                        'equity_before': current_equity,
                        'equity_after': current_equity,
                        'risk_multiplier': position_multiplier
                    })

        # Check for exit signal
        elif exit_exec_signals.loc[date] == 1 and position_size > 0 and exit_exec_prices.loc[date] > 0:
            exit_price = _calculate_exit_price(exit_exec_prices.loc[date], config)

            # Calculate P&L
            pnl = position_size * current_equity * (exit_price - entry_price) / entry_price
            equity_before_exit = current_equity
            current_equity += pnl

            trades.append({
                'date': date,
                'type': 'exit',
                'price': exit_price,
                'quantity': position_size,
                'pnl': pnl,
                'equity_before': equity_before_exit,
                'equity_after': current_equity,
                'risk_multiplier': position_multiplier
            })

            # Reset position
            position_size = 0.0
            entry_price = 0.0

        # Update position tracking
        positions.loc[date] = position_size

        # Track position reductions
        if position_multiplier < 1.0 and risk_manager is not None:
            risk_metrics['position_reductions'] += 1

    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    return positions, trades_df, risk_metrics


def _calculate_entry_price(price: float, config: Dict[str, Any]) -> float:
    """Calculate entry price with slippage."""
    slippage_pct = config.get('slippage_pct', 0.1) / 100.0
    commission_per_share = config.get('commission_per_share', 0.01)
    return price * (1 + slippage_pct) + commission_per_share


def _calculate_exit_price(price: float, config: Dict[str, Any]) -> float:
    """Calculate exit price with slippage."""
    slippage_pct = config.get('slippage_pct', 0.1) / 100.0
    commission_per_share = config.get('commission_per_share', 0.01)
    return price * (1 - slippage_pct) - commission_per_share


def _calculate_performance_metrics(
    data: pd.DataFrame,
    positions: pd.Series,
    trades: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    """
    if trades.empty:
        return {
            'total_return': 0.0,
            'cagr': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0
        }

    # Basic trade statistics
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] < 0]

    # CAGR (Compound Annual Growth Rate)
    start_equity = trades.iloc[0]['equity_before'] if not trades.empty else 10000.0
    end_equity = trades.iloc[-1]['equity_after'] if not trades.empty else 10000.0

    start_date = trades.iloc[0]['date'] if not trades.empty else data.index[0]
    end_date = trades.iloc[-1]['date'] if not trades.empty else data.index[-1]
    years = (end_date - start_date).days / 365.25

    if years > 0:
        cagr = (end_equity / start_equity) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Sharpe Ratio (annualized, risk-free rate = 4%)
    if not trades.empty and 'pnl' in trades.columns:
        returns = trades.set_index('date')['pnl'].pct_change().dropna()
        if len(returns) > 0:
            excess_returns = returns - 0.04/252  # Daily risk-free rate
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0.0
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Maximum Drawdown
    if not trades.empty:
        equity_curve = trades.set_index('date')['equity_after'].reindex(data.index).fillna(method='ffill')
        equity_curve = equity_curve.fillna(10000.0)
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0.0

    # Win Rate and other metrics
    total_trades = len(trades) // 2  # Each round trip = 2 trades
    winning_trades_count = len(winning_trades)
    win_rate = winning_trades_count / total_trades if total_trades > 0 else 0.0

    avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0.0
    avg_loss = abs(losing_trades['pnl'].mean()) if not losing_trades.empty else 0.0

    gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0.0
    gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calmar Ratio (CAGR / Max Drawdown)
    calmar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else float('inf')

    return {
        'total_return': (end_equity / start_equity) - 1,
        'cagr': cagr,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'calmar_ratio': calmar_ratio
    }
