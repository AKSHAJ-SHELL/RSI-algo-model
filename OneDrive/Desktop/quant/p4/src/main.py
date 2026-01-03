#!/usr/bin/env python3
"""
CLI entry point for RSI trading scanner.

Usage:
    python -m src.main --help
"""

import argparse
from datetime import datetime

from src.core.backtest import run_backtest_engine
from src.trading.scanner import scan_tickers
from src.utils.config import get_default_config
from src.data.validators import validate_ticker_list
from src.utils.logging import setup_logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RSI Mean-Reversion Trading Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest single ticker
  python -m src.main backtest --ticker SPY --start 2023-01-01 --end 2025-12-31

  # Scan for signals
  python -m src.main scan --tickers SPY,QQQ

  # Run web interface
  python -m src.main web
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest analysis')
    backtest_parser.add_argument('--ticker', required=True, help='Stock ticker symbol')
    backtest_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for current signals')
    scan_parser.add_argument('--tickers', required=True, help='Comma-separated ticker symbols')

    # Web command
    web_parser = subparsers.add_parser('web', help='Run web interface')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    setup_logging()

    # Load config
    config = get_default_config()

    try:
        if args.command == 'backtest':
            return handle_backtest(args, config)
        elif args.command == 'scan':
            return handle_scan(args, config)
        elif args.command == 'web':
            return handle_web()
    except Exception as e:
        print(f"Error: {e}")
        return 1


def handle_backtest(args, config):
    """Handle backtest command."""
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
    except ValueError:
        print("Error: Invalid date format. Use YYYY-MM-DD")
        return 1

    print(f"Running backtest for {args.ticker} from {start_date} to {end_date}...")

    try:
        # Import here to avoid circular imports
        from src.orchestrators.backtest import run_backtest

        results = run_backtest(args.ticker, start_date, end_date, config)

        if results and results.get('success'):
            display_backtest_results(results)
        else:
            print("Backtest failed")
            return 1

    except Exception as e:
        print(f"Backtest error: {e}")
        return 1

    return 0


def handle_scan(args, config):
    """Handle scan command."""
    try:
        tickers = validate_ticker_list([t.strip() for t in args.tickers.split(',')])
    except Exception as e:
        print(f"Error: Invalid ticker format: {e}")
        return 1

    print(f"Scanning {len(tickers)} tickers...")

    signals = scan_tickers(tickers, config)

    if signals:
        print("\nScan Results:")
        for signal in signals:
            print(f"{signal['ticker']}: {signal['signal']} - RSI: {signal.get('rsi_value', 'N/A')}")
    else:
        print("No signals found")

    return 0


def handle_web():
    """Handle web command."""
    print("Starting web interface...")
    try:
        from src.web.app import main as web_main
        web_main()
    except Exception as e:
        print(f"Error starting web interface: {e}")
        return 1

    return 0


def display_backtest_results(results):
    """Display backtest results."""
    metrics = results.get('metrics', {})

    print("
Backtest Results:")
    print(f"CAGR: {metrics.get('cagr', 0):.1%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Total Return: {metrics.get('total_return', 0):.1%}")

    if results.get('warnings'):
        print("
Warnings:")
        for warning in results['warnings']:
            print(f"  ⚠️ {warning}")


if __name__ == "__main__":
    exit(main())
