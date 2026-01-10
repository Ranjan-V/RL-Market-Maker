"""
Analyze live trading results
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def analyze_trading_log(log_file: str):
    """Analyze a trading log file"""
    
    # Load data
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    steps = pd.DataFrame(data['steps'])
    
    print("=" * 60)
    print("LIVE TRADING ANALYSIS")
    print("=" * 60)
    print()
    
    # Basic info
    print("Session Info:")
    print(f"  Symbol: {metadata['symbol']}")
    print(f"  Model: {metadata['model_type']}")
    print(f"  Duration: {metadata.get('total_steps', len(steps))} steps")
    print(f"  Start: {metadata['start_time']}")
    print(f"  End: {metadata['end_time']}")
    print()
    
    # Performance metrics
    print("Performance:")
    print(f"  Final PnL: ${steps['pnl'].iloc[-1]:,.2f}")
    print(f"  Max PnL: ${steps['pnl'].max():,.2f}")
    print(f"  Min PnL: ${steps['pnl'].min():,.2f}")
    print(f"  Max Drawdown: ${steps['pnl'].min() - steps['pnl'].max():,.2f}")
    print()
    
    # Trading activity
    total_trades = sum(len(step) for step in steps['orders'])
    print("Trading Activity:")
    print(f"  Total orders placed: {total_trades}")
    print(f"  Avg orders per step: {total_trades / len(steps):.1f}")
    print(f"  Final inventory: {steps['inventory'].iloc[-1]:+.6f} BTC")
    print(f"  Max inventory: {steps['inventory'].abs().max():.6f} BTC")
    print()
    
    # Market conditions
    print("Market Conditions:")
    print(f"  Start price: ${steps['mid_price'].iloc[0]:,.2f}")
    print(f"  End price: ${steps['mid_price'].iloc[-1]:,.2f}")
    print(f"  Price change: {(steps['mid_price'].iloc[-1] / steps['mid_price'].iloc[0] - 1) * 100:+.2f}%")
    print(f"  Avg spread: {steps['spread_bps'].mean():.2f} bps")
    print()
    
    # Create visualizations
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: PnL over time
    axes[0].plot(steps['step'], steps['pnl'], linewidth=2)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_title('Profit & Loss Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('PnL ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Inventory over time
    axes[1].plot(steps['step'], steps['inventory'], linewidth=2, color='orange')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title('Inventory Position Over Time', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Inventory (BTC)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Mid price over time
    axes[2].plot(steps['step'], steps['mid_price'], linewidth=2, color='green')
    axes[2].set_title('BTC Price Over Time', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Price ($)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = Path(log_file).with_suffix('.png')
    plt.savefig(plot_file, dpi=150)
    print(f"✓ Plot saved to: {plot_file}")
    
    plt.show()
    
    # Calculate Sharpe ratio (if enough data)
    if len(steps) > 10:
        returns = steps['pnl'].diff()
        sharpe = returns.mean() / (returns.std() + 1e-6) * np.sqrt(len(steps))
        print(f"\nSharpe Ratio: {sharpe:.3f}")
    
    return steps


def compare_sessions(log_dir: str = "data/logs"):
    """Compare multiple trading sessions"""
    log_files = list(Path(log_dir).glob("*.json"))
    
    if not log_files:
        print("No log files found!")
        return
    
    print("=" * 60)
    print(f"FOUND {len(log_files)} TRADING SESSIONS")
    print("=" * 60)
    print()
    
    results = []
    
    for log_file in sorted(log_files):
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        metadata = data['metadata']
        steps = pd.DataFrame(data['steps'])
        
        results.append({
            'file': log_file.name,
            'model': metadata['model_type'],
            'symbol': metadata['symbol'],
            'duration_min': len(steps) / 6,  # Assuming 10s intervals
            'final_pnl': steps['pnl'].iloc[-1],
            'max_pnl': steps['pnl'].max(),
            'trades': sum(len(step) for step in steps['orders'])
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    
    # Best session
    best = df.loc[df['final_pnl'].idxmax()]
    print(f"🏆 Best Session: {best['file']}")
    print(f"   Final PnL: ${best['final_pnl']:,.2f}")
    print(f"   Model: {best['model']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Analyze specific log file
        log_file = sys.argv[1]
        analyze_trading_log(log_file)
    else:
        # Compare all sessions
        compare_sessions()