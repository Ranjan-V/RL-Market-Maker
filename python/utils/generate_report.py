"""
Generate comprehensive performance report
"""
import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from backtesting.backtest import Backtester
from backtesting.metrics import PerformanceMetrics
from env.market_env import MarketMakerEnv
from baselines.avellaneda_stoikov import AvellanedaStoikovAgent
from stable_baselines3 import SAC, PPO


def generate_full_report(
    ppo_model_path: str,
    sac_model_path: str,
    output_dir: str = "reports"
):
    """
    Generate complete performance report
    
    Args:
        ppo_model_path: Path to PPO model
        sac_model_path: Path to SAC model
        output_dir: Where to save report
    """
    print("=" * 60)
    print("GENERATING COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 60)
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_path / f"report_{timestamp}"
    report_dir.mkdir(exist_ok=True)
    
    # Create environment
    env = MarketMakerEnv()
    
    # Load models
    print("Loading models...")
    ppo_agent = PPO.load(ppo_model_path)
    sac_agent = SAC.load(sac_model_path)
    as_agent = AvellanedaStoikovAgent(risk_aversion=0.1)
    print("✓ Models loaded")
    print()
    
    # Backtest all agents
    backtester = Backtester(env, n_episodes=100, verbose=True)
    
    agents = {
        'PPO Agent': ppo_agent,
        'SAC Agent': sac_agent,
        'AS Baseline': as_agent
    }
    
    df_comparison, all_results = backtester.compare_agents(agents)
    
    # Save comparison table
    df_comparison.to_csv(report_dir / "comparison_table.csv", index=False)
    print(f"✓ Saved comparison table")
    
    # Calculate detailed metrics for each agent
    print("\nCalculating detailed metrics...")
    detailed_metrics = {}
    
    for result in all_results:
        name = result['agent_name']
        pnls = result['pnls']
        metrics = PerformanceMetrics.get_all_metrics(pnls)
        detailed_metrics[name] = metrics
        
        # Print metrics
        PerformanceMetrics.print_metrics(metrics, name)
        
        # Save metrics to JSON
        with open(report_dir / f"{name.replace(' ', '_')}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Performance comparison plot
    results_dict = {r['agent_name']: r['pnls'] for r in all_results}
    fig = PerformanceMetrics.plot_performance(
        results_dict,
        title="Market Making Strategy Comparison"
    )
    fig.savefig(report_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Saved performance comparison plot")
    
    # Individual PnL trajectories
    fig, axes = plt.subplots(len(all_results), 1, figsize=(12, 4 * len(all_results)))
    if len(all_results) == 1:
        axes = [axes]
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        pnls = result['pnls']
        cumulative = pd.Series(pnls).cumsum()
        
        ax.plot(cumulative, linewidth=2, color=f'C{idx}')
        ax.fill_between(range(len(cumulative)), cumulative, 0, alpha=0.3, color=f'C{idx}')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f"{result['agent_name']} - Cumulative PnL", fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = (
            f"Final PnL: ${cumulative.iloc[-1]:,.2f}\n"
            f"Sharpe: {detailed_metrics[result['agent_name']]['sharpe_ratio']:.3f}\n"
            f"Max DD: {detailed_metrics[result['agent_name']]['max_drawdown']:.2%}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(report_dir / "individual_trajectories.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Saved individual trajectory plots")
    
    # Create summary report (text file)
    with open(report_dir / "summary_report.txt", 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MARKET MAKING PERFORMANCE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Episodes: {backtester.n_episodes}\n\n")
        
        f.write("COMPARISON SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(df_comparison.to_string(index=False))
        f.write("\n\n")
        
        # Winner
        best_idx = df_comparison['Mean PnL'].idxmax()
        winner = df_comparison.loc[best_idx, 'Agent']
        winner_pnl = df_comparison.loc[best_idx, 'Mean PnL']
        
        f.write(f"WINNER: {winner}\n")
        f.write(f"Mean PnL: ${winner_pnl:,.2f}\n\n")
        
        # Detailed metrics for winner
        f.write("DETAILED METRICS (WINNER)\n")
        f.write("-" * 60 + "\n")
        winner_metrics = detailed_metrics[winner]
        for key, value in winner_metrics.items():
            if isinstance(value, float):
                if '%' in key or 'ratio' in key.lower():
                    f.write(f"{key:20s}: {value:>12.3f}\n")
                else:
                    f.write(f"{key:20s}: ${value:>12,.2f}\n")
        
        f.write("\n")
    
    print("✓ Saved summary report")
    
    print()
    print("=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"Report saved to: {report_dir}")
    print()
    print("Generated files:")
    print("  - comparison_table.csv")
    print("  - performance_comparison.png")
    print("  - individual_trajectories.png")
    print("  - summary_report.txt")
    print("  - *_metrics.json (for each agent)")
    print()
    
    return report_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate performance report')
    parser.add_argument('--ppo', type=str, required=True, help='Path to PPO model')
    parser.add_argument('--sac', type=str, required=True, help='Path to SAC model')
    parser.add_argument('--output', type=str, default='reports', help='Output directory')
    
    args = parser.parse_args()
    
    generate_full_report(args.ppo, args.sac, args.output)