"""
Performance Metrics for Trading Strategies
"""
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceMetrics:
    """Calculate comprehensive trading performance metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-6) * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        if len(returns) < 2:
            return 0.0
        cumulative = np.cumsum(returns)
        max_dd = abs(PerformanceMetrics.calculate_max_drawdown(cumulative))
        if max_dd == 0:
            return 0.0
        return np.mean(returns) * 252 / max_dd
    
    @staticmethod
    def calculate_win_rate(returns: np.ndarray) -> float:
        """Calculate win rate (% of positive returns)"""
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)
    
    @staticmethod
    def calculate_profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return np.inf if profits > 0 else 0.0
        return profits / losses
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        var = PerformanceMetrics.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def get_all_metrics(pnls: List[float]) -> Dict:
        """Calculate all performance metrics"""
        pnls_array = np.array(pnls)
        returns = np.diff(pnls_array) if len(pnls_array) > 1 else pnls_array
        cumulative = np.cumsum(pnls_array)
        
        metrics = {
            'total_pnl': pnls_array[-1] if len(pnls_array) > 0 else 0.0,
            'mean_pnl': np.mean(pnls_array),
            'std_pnl': np.std(pnls_array),
            'median_pnl': np.median(pnls_array),
            'min_pnl': np.min(pnls_array) if len(pnls_array) > 0 else 0.0,
            'max_pnl': np.max(pnls_array) if len(pnls_array) > 0 else 0.0,
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(returns),
            'max_drawdown': PerformanceMetrics.calculate_max_drawdown(cumulative),
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(returns),
            'win_rate': PerformanceMetrics.calculate_win_rate(returns),
            'profit_factor': PerformanceMetrics.calculate_profit_factor(returns),
            'var_95': PerformanceMetrics.calculate_var(returns, 0.95),
            'cvar_95': PerformanceMetrics.calculate_cvar(returns, 0.95)
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict, name: str = "Strategy"):
        """Print formatted metrics"""
        print("=" * 60)
        print(f"PERFORMANCE METRICS: {name}")
        print("=" * 60)
        print()
        
        print("Return Metrics:")
        print(f"  Total PnL:        ${metrics['total_pnl']:>12,.2f}")
        print(f"  Mean PnL:         ${metrics['mean_pnl']:>12,.2f}")
        print(f"  Median PnL:       ${metrics['median_pnl']:>12,.2f}")
        print(f"  Std Dev:          ${metrics['std_pnl']:>12,.2f}")
        print(f"  Range:            ${metrics['min_pnl']:>12,.2f} to ${metrics['max_pnl']:,.2f}")
        print()
        
        print("Risk-Adjusted Returns:")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>12.3f}")
        print(f"  Sortino Ratio:    {metrics['sortino_ratio']:>12.3f}")
        print(f"  Calmar Ratio:     {metrics['calmar_ratio']:>12.3f}")
        print()
        
        print("Risk Metrics:")
        print(f"  Max Drawdown:     {metrics['max_drawdown']:>12.2%}")
        print(f"  VaR (95%):        ${metrics['var_95']:>12,.2f}")
        print(f"  CVaR (95%):       ${metrics['cvar_95']:>12,.2f}")
        print()
        
        print("Win Metrics:")
        print(f"  Win Rate:         {metrics['win_rate']:>12.2%}")
        print(f"  Profit Factor:    {metrics['profit_factor']:>12.2f}")
        print()
        print("=" * 60)
    
    @staticmethod
    def plot_performance(
        results_dict: Dict[str, List[float]],
        title: str = "Strategy Comparison"
    ):
        """
        Plot performance comparison
        
        Args:
            results_dict: {strategy_name: [pnls]}
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Cumulative PnL
        ax = axes[0, 0]
        for name, pnls in results_dict.items():
            cumulative = np.cumsum(pnls)
            ax.plot(cumulative, label=name, linewidth=2)
        ax.set_title('Cumulative PnL', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 2: PnL Distribution
        ax = axes[0, 1]
        data_to_plot = [pnls for pnls in results_dict.values()]
        labels = list(results_dict.keys())
        ax.violinplot(data_to_plot, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title('PnL Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('PnL ($)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 3: Rolling Sharpe Ratio
        ax = axes[1, 0]
        window = 20
        for name, pnls in results_dict.items():
            if len(pnls) > window:
                returns = np.diff(pnls)
                rolling_sharpe = pd.Series(returns).rolling(window).apply(
                    lambda x: np.mean(x) / (np.std(x) + 1e-6) * np.sqrt(252)
                )
                ax.plot(rolling_sharpe, label=name, linewidth=2)
        ax.set_title(f'Rolling Sharpe Ratio (window={window})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 4: Drawdown
        ax = axes[1, 1]
        for name, pnls in results_dict.items():
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / (running_max + 1e-6)
            ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, label=name)
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Simulate some PnL data
    np.random.seed(42)
    
    strategy1_pnls = np.cumsum(np.random.randn(100) * 10 + 0.5)
    strategy2_pnls = np.cumsum(np.random.randn(100) * 8 + 0.3)
    
    # Calculate metrics
    metrics1 = PerformanceMetrics.get_all_metrics(strategy1_pnls.tolist())
    metrics2 = PerformanceMetrics.get_all_metrics(strategy2_pnls.tolist())
    
    # Print
    PerformanceMetrics.print_metrics(metrics1, "Strategy 1")
    PerformanceMetrics.print_metrics(metrics2, "Strategy 2")
    
    # Plot
    results = {
        'Strategy 1': strategy1_pnls.tolist(),
        'Strategy 2': strategy2_pnls.tolist()
    }
    
    fig = PerformanceMetrics.plot_performance(results)
    plt.show()