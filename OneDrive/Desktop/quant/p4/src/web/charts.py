"""
Chart visualization for RSI trading scanner.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional


def create_price_rsi_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create price chart with RSI overlay.

    Args:
        data: DataFrame with OHLCV data and RSI
        ticker: Stock ticker

    Returns:
        Plotly figure
    """
    if data.empty or 'RSI_14' not in data.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=(f'{ticker} Price', 'RSI(14)'),
                       vertical_spacing=0.1, row_width=[0.7, 0.3])

    # Price chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ), row=1, col=1)

    # RSI chart
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI_14'],
        name='RSI(14)',
        line=dict(color='purple')
    ), row=2, col=1)

    # RSI levels
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)

    fig.update_layout(height=600, title=f"{ticker} Price & RSI Analysis")
    return fig


def create_equity_curve_chart(equity_curve: pd.Series, title: str = "Equity Curve") -> go.Figure:
    """
    Create equity curve chart.

    Args:
        equity_curve: Series with equity values
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='Equity',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=400
    )

    return fig


def create_drawdown_chart(equity_curve: pd.Series) -> go.Figure:
    """
    Create drawdown chart.

    Args:
        equity_curve: Series with equity values

    Returns:
        Plotly figure
    """
    # Calculate drawdown
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='red'),
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))

    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=300
    )

    return fig
