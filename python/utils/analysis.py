"""
Analysis utilities
"""
import numpy as np
import pandas as pd
from typing import Dict, List


def analyze_performance(pnls: List[float]) -> Dict:
    """
    Analyze performance from PnL list
    
    Args:
        pnls: List of PnL values
        
    Returns:
        Dictionary of performance metrics
    """
    return {
        'mean': np.mean(pnls),
        'std': np.std(pnls),
        'sharpe': np.mean(pnls) / (np.std(pnls) + 1e-6),
        'max': np.max(pnls),
        'min': np.min(pnls)
    }


def calculate_drawdown(cumulative_pnl: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = (cumulative_pnl - running_max) / (running_max + 1e-6)
    return np.min(drawdown)