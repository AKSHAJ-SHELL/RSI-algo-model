#!/usr/bin/env python3
"""
RSI Mean-Reversion Scanner - Streamlit Web Application

A clean, focused trading scanner for RSI-based strategies.
"""

# Import from new clean structure
from src.web.app import main as web_main


# Run the web application
if __name__ == "__main__":
    web_main()
