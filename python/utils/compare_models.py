"""
Compare saved models against baseline
"""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, SAC
from env.market_env import MarketMakerEnv
from baselines.avellaneda_stoikov import AvellanedaStoikovAgent, evaluate_agent


def evaluate_saved_model(model_path: str, n_episodes: int = 20):
    """Evaluate a saved model"""
    env = MarketMakerEnv()
    
    # Load model
    try:
        if 'ppo' in model_path.lower():
            model = PPO.load(model_path)
        elif 'sac' in model_path.lower():
            model = SAC.load(model_path)
        else:
            print(f"Cannot determine model type from path: {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    episode_pnls = []
    episode_inventories = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        inv_history = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            inv_history.append(abs(info['inventory']))
        
        episode_pnls.append(info['total_pnl'])
        episode_inventories.append(np.mean(inv_history))
    
    results = {
        'mean_pnl': np.mean(episode_pnls),
        'std_pnl': np.std(episode_pnls),
        'sharpe_ratio': np.mean(episode_pnls) / (np.std(episode_pnls) + 1e-6),
        'mean_inventory': np.mean(episode_inventories),
        'max_pnl': np.max(episode_pnls),
        'min_pnl': np.min(episode_pnls)
    }
    
    return results


def main():
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print()
    
    # Evaluate AS baseline
    print("Evaluating Avellaneda-Stoikov baseline...")
    as_agent = AvellanedaStoikovAgent(risk_aversion=0.1)
    as_env = MarketMakerEnv()
    as_results = evaluate_agent(as_agent, as_env, n_episodes=20)
    
    print(f"AS Baseline:")
    print(f"  Mean PnL: ${as_results['mean_pnl']:.2f} ± ${as_results['std_pnl']:.2f}")
    print(f"  Sharpe:   {as_results['sharpe_ratio']:.3f}")
    print()
    
    # Find all saved models
    models_dir = Path("logs/tensorboard")
    
    # Check for PPO best model
    ppo_dirs = list(models_dir.glob("ppo_*/best_model/best_model.zip"))
    if ppo_dirs:
        print("=" * 60)
        print("PPO BEST MODEL")
        print("=" * 60)
        
        # Use most recent
        ppo_best = sorted(ppo_dirs)[-1]
        print(f"Model: {ppo_best}")
        print()
        
        ppo_results = evaluate_saved_model(str(ppo_best), n_episodes=20)
        
        if ppo_results:
            print(f"PPO Best Model:")
            print(f"  Mean PnL: ${ppo_results['mean_pnl']:.2f} ± ${ppo_results['std_pnl']:.2f}")
            print(f"  Sharpe:   {ppo_results['sharpe_ratio']:.3f}")
            print(f"  Max PnL:  ${ppo_results['max_pnl']:.2f}")
            print(f"  Min PnL:  ${ppo_results['min_pnl']:.2f}")
            
            improvement = ppo_results['mean_pnl'] - as_results['mean_pnl']
            print()
            print(f"Improvement over AS: ${improvement:.2f} ({improvement/abs(as_results['mean_pnl'])*100:.1f}%)")
            print()
    
    # Check for SAC best model
    sac_dirs = list(models_dir.glob("sac_*/best_model/best_model.zip"))
    if sac_dirs:
        print("=" * 60)
        print("SAC BEST MODEL")
        print("=" * 60)
        
        sac_best = sorted(sac_dirs)[-1]
        print(f"Model: {sac_best}")
        print()
        
        sac_results = evaluate_saved_model(str(sac_best), n_episodes=20)
        
        if sac_results:
            print(f"SAC Best Model:")
            print(f"  Mean PnL: ${sac_results['mean_pnl']:.2f} ± ${sac_results['std_pnl']:.2f}")
            print(f"  Sharpe:   {sac_results['sharpe_ratio']:.3f}")
            print(f"  Max PnL:  ${sac_results['max_pnl']:.2f}")
            print(f"  Min PnL:  ${sac_results['min_pnl']:.2f}")
            
            improvement = sac_results['mean_pnl'] - as_results['mean_pnl']
            print()
            print(f"Improvement over AS: ${improvement:.2f} ({improvement/abs(as_results['mean_pnl'])*100:.1f}%)")
            print()
    
    # Final comparison
    if ppo_dirs or sac_dirs:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"AS Baseline: ${as_results['mean_pnl']:.2f}")
        
        if ppo_dirs and ppo_results:
            print(f"PPO Best:    ${ppo_results['mean_pnl']:.2f}")
        
        if sac_dirs and sac_results:
            print(f"SAC Best:    ${sac_results['mean_pnl']:.2f}")
        
        print()
        
        # Determine winner
        all_results = [('AS', as_results['mean_pnl'])]
        if ppo_dirs and ppo_results:
            all_results.append(('PPO', ppo_results['mean_pnl']))
        if sac_dirs and sac_results:
            all_results.append(('SAC', sac_results['mean_pnl']))
        
        winner = max(all_results, key=lambda x: x[1])
        print(f"🏆 Winner: {winner[0]} with ${winner[1]:.2f} mean PnL")
    else:
        print("No trained models found. Train PPO or SAC first!")


if __name__ == "__main__":
    main()