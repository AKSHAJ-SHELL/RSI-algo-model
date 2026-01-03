"""
Streamlit web application for RSI trading scanner.
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta

from src.core.backtest import run_backtest_engine
from src.trading.scanner import scan_tickers
from src.utils.config import get_default_config, create_config_from_presets
from src.data.validators import sanitize_ticker_list
from src.utils.logging import get_logger

logger = get_logger("web_app")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RSI Trading Scanner",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 RSI Mean-Reversion Scanner")

    # Disclaimer
    with st.expander("⚠️ Important Disclaimer", expanded=False):
        st.warning("""
        **Educational & Personal Use Only**

        This tool is for educational and personal use only. Not financial advice.
        Past performance ≠ future results. Trading involves substantial risk of loss.

        **RSI Limitations:** RSI mean-reversion works best in range-bound markets but
        fails miserably in strong trends. This strategy has historically underperformed
        buy-and-hold in trending markets (2026 may have trends).
        """)

    # Sidebar configuration
    config = render_sidebar()

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Live Scanner", "📊 Backtest", "📈 Charts"])

    with tab1:
        render_live_scanner(config)

    with tab2:
        render_backtest(config)

    with tab3:
        render_charts()


def render_sidebar():
    """Render sidebar configuration."""
    st.sidebar.header("⚙️ Configuration")

    # Preset selection
    preset = st.sidebar.selectbox(
        "Strategy Preset",
        ["Custom", "Conservative", "Moderate", "Aggressive"],
        help="Choose a preset strategy or customize below"
    )

    config = get_default_config()

    if preset != "Custom":
        config.update(create_config_from_presets(preset))

        st.sidebar.success(f"✅ {preset} preset loaded")

        # Allow overrides
        if st.sidebar.checkbox("Customize Settings"):
            config = render_config_options(config)
    else:
        config = render_config_options(config)

    return config


def render_config_options(config):
    """Render configuration options."""
    st.sidebar.subheader("Entry Settings")
    config['oversold_threshold'] = st.sidebar.slider(
        "Oversold Threshold", 20, 40, int(config['oversold_threshold']),
        help="RSI level to trigger buy signals"
    )

    st.sidebar.subheader("Risk Settings")
    config['position_size_pct'] = st.sidebar.slider(
        "Position Size %", 0.1, 5.0, config['position_size_pct'],
        help="Position size as % of capital"
    )

    config['max_hold_days'] = st.sidebar.slider(
        "Max Hold Days", 10, 120, config['max_hold_days'],
        help="Maximum days to hold a position"
    )

    return config


def render_live_scanner(config):
    """Render live scanner interface."""
    st.header("🔍 Live Scanner")

    # Ticker input
    ticker_input = st.text_area(
        "Tickers to Scan",
        placeholder="AAPL\nMSFT\nGOOGL",
        help="Enter ticker symbols, one per line"
    )

    if st.button("🔍 Scan Tickers", type="primary"):
        if not ticker_input.strip():
            st.error("Please enter at least one ticker")
            return

        tickers = sanitize_ticker_list(ticker_input)

        with st.spinner("Scanning tickers..."):
            signals = scan_tickers(tickers, config)

        if signals:
            # Display results
            df = pd.DataFrame(signals)

            # Color coding
            def color_signal(val):
                if val == 'BUY':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'SELL':
                    return 'background-color: #f8d7da; color: #721c24'
                return ''

            styled_df = df.style.apply(
                lambda x: [color_signal(val) for val in x],
                subset=['signal']
            )

            st.dataframe(styled_df, use_container_width=True)

            # Summary stats
            buy_signals = len(df[df['signal'] == 'BUY'])
            st.info(f"📊 Found {buy_signals} BUY signals out of {len(signals)} tickers scanned")

        else:
            st.info("No signals found or scanning failed")


def render_backtest(config):
    """Render backtest interface."""
    st.header("📊 Backtest")

    col1, col2 = st.columns(2)

    with col1:
        ticker = st.text_input("Ticker", "SPY", help="Stock ticker to backtest")
        start_date = st.date_input("Start Date", date.today() - timedelta(days=365))

    with col2:
        end_date = st.date_input("End Date", date.today())
        enable_out_of_sample = st.checkbox("Out-of-Sample Testing", help="Test on unseen data")

    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                # Import here to avoid circular imports
                from src.orchestrators.backtest import run_backtest

                config['enable_out_of_sample'] = enable_out_of_sample
                results = run_backtest(ticker, start_date, end_date, config)

                if results:
                    display_backtest_results(results)
                else:
                    st.error("Backtest failed - check logs for details")

            except Exception as e:
                st.error(f"Backtest error: {str(e)}")
                logger.error(f"Backtest failed: {e}")


def display_backtest_results(results):
    """Display backtest results."""
    metrics = results.get('metrics', {})

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{metrics.get('total_return', 0):.1%}")

    with col2:
        st.metric("CAGR", f"{metrics.get('cagr', 0):.1%}")

    with col3:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}")

    with col4:
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")

    # Performance details
    with st.expander("📈 Detailed Metrics", expanded=True):
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df.T, use_container_width=True)

    # Warnings
    if results.get('warnings'):
        with st.expander("⚠️ Warnings", expanded=True):
            for warning in results['warnings']:
                st.warning(warning)


def render_charts():
    """Render charts interface."""
    st.header("📈 Charts")
    st.info("Chart visualization will be implemented here")
    # Placeholder for charts functionality


if __name__ == "__main__":
    main()
