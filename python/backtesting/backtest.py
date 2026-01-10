"""
Backtesting Engine for Market Making Strategies
Supports both Python and C++ order books
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

sys.path.append(str(Path(__file__).parent.parent))

from env.market_env import MarketMakerEnv
from stable_baselines3 import PPO, SAC
from baselines.avellaneda_stoikov import AvellanedaStoikovAgent


class Backtester:
    """
    Comprehensive backtesting engine
    """
    
    def __init__(
        self,
        env: MarketMakerEnv,
        n_episodes: int = 100,
        verbose: bool = True,
        use_cpp: bool = False
    ):
        """
        Args:
            env: Market environment
            n_episodes: Number of episodes to run
            verbose: Print progress
            use_cpp: Use C++ order book (if available)
        """
        self.env = env
        self.n_episodes = n_episodes
        self.verbose = verbose
        self.use_cpp = use_cpp
        
        if use_cpp:
            try:
                import fast_orderbook
                self.cpp_available = True
                print("✓ Using C++ order book for speed")
            except ImportError:
                self.cpp_available = False
                print("⚠ C++ order book not available, using Python")
    
    def run_agent(
        self,
        agent,
        agent_name: str = "Agent",
        deterministic: bool = True
    ) -> Dict:
        """
        Run agent on environment
        
        Args:
            agent: Agent to test (RL model or baseline)
            agent_name: Name for logging
            deterministic: Use deterministic policy
        
        Returns:
            Dictionary of results
        """
        if self.verbose:
            print(f"\nBacktesting {agent_name}...")
            print(f"Episodes: {self.n_episodes}")
        
        episode_results = []
        start_time = time.time()
        
        for episode in range(self.n_episodes):
            obs, info = self.env.reset()
            done = False
            
            episode_data = {
                'pnls': [],
                'inventories': [],
                'spreads': [],
                'trades': 0
            }
            
            while not done:
                # Get action from agent
                if hasattr(agent, 'predict'):
                    # RL agent
                    action, _ = agent.predict(obs, deterministic=deterministic)
                elif hasattr(agent, 'get_action'):
                    # Baseline agent
                    action = agent.get_action(obs, self.env)
                else:
                    raise ValueError("Agent must have predict() or get_action() method")
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Record metrics
                episode_data['pnls'].append(info['total_pnl'])
                episode_data['inventories'].append(abs(info['inventory']))
            
            # Episode summary
            episode_results.append({
                'episode': episode,
                'final_pnl': info['total_pnl'],
                'avg_inventory': np.mean(episode_data['inventories']),
                'max_inventory': np.max(episode_data['inventories']),
                'steps': self.env.current_step
            })
            
            if self.verbose and (episode + 1) % 10 == 0:
                avg_pnl = np.mean([r['final_pnl'] for r in episode_results])
                print(f"  Episode {episode + 1}/{self.n_episodes}: Avg PnL = ${avg_pnl:.2f}")
        
        elapsed = time.time() - start_time
        
        # Calculate statistics
        pnls = [r['final_pnl'] for r in episode_results]
        
        results = {
            'agent_name': agent_name,
            'n_episodes': self.n_episodes,
            'mean_pnl': np.mean(pnls),
            'std_pnl': np.std(pnls),
            'median_pnl': np.median(pnls),
            'min_pnl': np.min(pnls),
            'max_pnl': np.max(pnls),
            'sharpe_ratio': np.mean(pnls) / (np.std(pnls) + 1e-6),
            'win_rate': np.sum(np.array(pnls) > 0) / len(pnls),
            'avg_inventory': np.mean([r['avg_inventory'] for r in episode_results]),
            'elapsed_time': elapsed,
            'episodes_per_sec': self.n_episodes / elapsed,
            'episode_results': episode_results,
            'pnls': pnls
        }
        
        if self.verbose:
            self._print_results(results)
        
        return results
    
    def compare_agents(self, agents: Dict) -> pd.DataFrame:
        """
        Compare multiple agents
        
        Args:
            agents: Dict of {name: agent} pairs
        
        Returns:
            DataFrame with comparison
        """
        all_results = []
        
        print("=" * 60)
        print("BACKTESTING COMPARISON")
        print("=" * 60)
        
        for name, agent in agents.items():
            results = self.run_agent(agent, agent_name=name)
            all_results.append(results)
        
        # Create comparison DataFrame
        df = pd.DataFrame([{
            'Agent': r['agent_name'],
            'Mean PnL': r['mean_pnl'],
            'Std PnL': r['std_pnl'],
            'Sharpe': r['sharpe_ratio'],
            'Win Rate': r['win_rate'],
            'Max PnL': r['max_pnl'],
            'Min PnL': r['min_pnl'],
            'Avg Inventory': r['avg_inventory'],
            'Speed (eps/s)': r['episodes_per_sec']
        } for r in all_results])
        
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(df.to_string(index=False))
        print()
        
        # Determine winner
        best_idx = df['Mean PnL'].idxmax()
        best_agent = df.loc[best_idx, 'Agent']
        best_pnl = df.loc[best_idx, 'Mean PnL']
        
        print(f"🏆 Winner: {best_agent} with ${best_pnl:.2f} mean PnL")
        print()
        
        return df, all_results
    
    def _print_results(self, results: Dict):
        """Print formatted results"""
        print()
        print("-" * 60)
        print(f"Results for {results['agent_name']}")
        print("-" * 60)
        print(f"Mean PnL:     ${results['mean_pnl']:>10.2f} ± ${results['std_pnl']:.2f}")
        print(f"Median PnL:   ${results['median_pnl']:>10.2f}")
        print(f"Range:        ${results['min_pnl']:>10.2f} to ${results['max_pnl']:.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:>10.3f}")
        print(f"Win Rate:     {results['win_rate']:>10.1%}")
        print(f"Avg Inv:      {results['avg_inventory']:>10.4f}")
        print(f"Time:         {results['elapsed_time']:>10.1f}s ({results['episodes_per_sec']:.1f} eps/s)")
        print("-" * 60)


def quick_backtest(model_path: str, model_type: str = "SAC", n_episodes: int = 50):
    """
    Quick backtest of a trained model vs baseline
    
    Args:
        model_path: Path to saved model
        model_type: 'SAC' or 'PPO'
        n_episodes: Number of episodes
    """
    print("=" * 60)
    print("QUICK BACKTEST")
    print("=" * 60)
    print()
    
    # Create environment
    env = MarketMakerEnv()
    
    # Load RL model
    print(f"Loading {model_type} model...")
    if model_type.upper() == "SAC":
        rl_agent = SAC.load(model_path)
    elif model_type.upper() == "PPO":
        rl_agent = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    print("✓ Model loaded")
    
    # Create baseline
    as_agent = AvellanedaStoikovAgent(risk_aversion=0.1)
    
    # Backtest
    backtester = Backtester(env, n_episodes=n_episodes)
    
    agents = {
        f'{model_type} Agent': rl_agent,
        'AS Baseline': as_agent
    }
    
    df, results = backtester.compare_agents(agents)
    
    return df, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest trading agents')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--type', type=str, default='SAC', choices=['SAC', 'PPO'])
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    
    args = parser.parse_args()
    
    if args.model:
        quick_backtest(args.model, args.type, args.episodes)
    else:
        print("Usage: python backtest.py --model path/to/model.zip --type SAC --episodes 50")