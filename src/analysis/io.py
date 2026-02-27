"""
Output formatting utilities for analysis scripts.

This module provides consistent formatting for console output across all analysis scripts.
"""

from datetime import datetime


def print_section_header(title: str, symbol: str = "=") -> None:
    """
    Print a formatted section header for better output organization.
    
    Args:
        title: The title text to display in the header
        symbol: The character to use for the border (default: "=")
    
    Example:
        >>> print_section_header("ANALYSIS COMPLETE")
        ================================================================================
                                    ANALYSIS COMPLETE                                  
        ================================================================================
    """
    width = 80
    print(f"\n{symbol * width}")
    print(f"{title.center(width)}")
    print(f"{symbol * width}")


def print_subsection(title: str) -> None:
    """
    Print a formatted subsection header.
    
    Args:
        title: The subsection title to display
    
    Example:
        >>> print_subsection("Configuration")
        ────────────────────────────────────────────────────────────────────────────────
          Configuration
        ────────────────────────────────────────────────────────────────────────────────
    """
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


def print_status(message: str, status: str = "INFO") -> None:
    """
    Print a status message with timestamp and status indicator.
    
    Args:
        message: The status message to display
        status: The status type (INFO, SUCCESS, ERROR, WARNING) (default: "INFO")
    
    Valid status values and their symbols:
        - INFO: ℹ
        - SUCCESS: ✓
        - ERROR: ✗
        - WARNING: ⚠
        - Any other value: •
    
    Example:
        >>> print_status("Processing complete", "SUCCESS")
        [14:23:45] ✓ Processing complete
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_symbols = {
        "INFO": "ℹ",
        "SUCCESS": "✓",
        "ERROR": "✗",
        "WARNING": "⚠"
    }
    symbol = status_symbols.get(status, "•")
    print(f"[{timestamp}] {symbol} {message}")
