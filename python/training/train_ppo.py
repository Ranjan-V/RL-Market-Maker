"""
PPO Training Script for Market Making
Trains PPO agent with evaluation against AS baseline
"""
import sys
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime
from typing import Dict, List
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from env.market_env import MarketMakerEnv
from baselines.avellaneda_stoikov import AvellanedaStoikovAgent, evaluate_agent


class TradingCallback(EvalCallback):
    """Custom callback with comparison to AS baseline"""
    
    def __init__(self, *args, as_agent=None, as_results=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.as_agent = as_agent
        self.as_baseline_pnl = as_results['mean_pnl'] if as_results else None
        self.best_rl_pnl = -np.inf
        self.evaluation_count = 0
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # Check if evaluation just happened
        if len(self.evaluations_results) > 0 and len(self.evaluations_results) > self.evaluation_count:
            self.evaluation_count = len(self.evaluations_results)
            current_mean_pnl = np.mean(self.evaluations_results[-1])
            
            # Compare to AS baseline
            if self.as_baseline_pnl is not None:
                improvement = current_mean_pnl - self.as_baseline_pnl
                beat_baseline = "✓" if improvement > 0 else "✗"
                print(f"\n{'='*60}")
                print(f"Evaluation #{self.evaluation_count} {beat_baseline}")
                print(f"RL Agent PnL:  ${current_mean_pnl:.2f}")
                print(f"AS Baseline:   ${self.as_baseline_pnl:.2f}")
                print(f"Improvement:   ${improvement:.2f} ({improvement/abs(self.as_baseline_pnl)*100:.1f}%)")
                
                if current_mean_pnl > self.best_rl_pnl:
                    self.best_rl_pnl = current_mean_pnl
                    print(f"🎯 New best PnL!")
                
                print(f"{'='*60}\n")
        
        return result


def make_env(rank: int, seed: int = 0):
    """Create a single environment"""
    def _init():
        env = MarketMakerEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo(config_path: str = "configs/training_config.yaml"):
    """
    Main training function
    
    Args:
        config_path: Path to training configuration
    """
    print("="*60)
    print("RL MARKET MAKER - PPO TRAINING")
    print("="*60)
    print()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seeds
    seed = config['seed']
    np.random.seed(seed)
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config['logging']['tensorboard_log']) / f"ppo_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(log_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    print(f"Logs will be saved to: {log_dir}")
    print()
    
    # Create vectorized environments
    n_envs = config['training']['n_envs']
    print(f"Creating {n_envs} parallel environments...")
    
    if n_envs == 1:
        env = DummyVecEnv([make_env(0, seed)])
        eval_env = DummyVecEnv([make_env(0, seed + 1000)])
    else:
        env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
        eval_env = DummyVecEnv([make_env(0, seed + 1000)])
    
    print("✓ Environments created")
    print()
    
    # Evaluate AS baseline
    print("Evaluating Avellaneda-Stoikov baseline...")
    as_agent = AvellanedaStoikovAgent(
        risk_aversion=config['baseline']['as_gamma']
    )
    as_env = MarketMakerEnv()
    as_results = evaluate_agent(as_agent, as_env, n_episodes=10)
    
    print(f"AS Baseline PnL: ${as_results['mean_pnl']:.2f} ± ${as_results['std_pnl']:.2f}")
    print(f"AS Sharpe: {as_results['sharpe_ratio']:.3f}")
    print()
    
    # Create PPO model
    print("Creating PPO model...")
    ppo_config = config['ppo']
    
    model = PPO(
        policy=ppo_config['policy'],
        env=env,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        use_sde=ppo_config['use_sde'],
        policy_kwargs={
            'net_arch': ppo_config['net_arch'],
            'activation_fn': torch.nn.ReLU if ppo_config['activation_fn'].lower() == 'relu' else torch.nn.Tanh
        },
        verbose=config['logging']['verbose'],
        seed=seed,
        tensorboard_log=str(log_dir)
    )
    
    print("✓ Model created")
    print(f"  Policy: {ppo_config['policy']}")
    print(f"  Network: {ppo_config['net_arch']}")
    print(f"  Learning rate: {ppo_config['learning_rate']}")
    print()
    
    # Setup callbacks
    print("Setting up callbacks...")
    
    # Evaluation callback
    eval_callback = TradingCallback(
        eval_env=eval_env,
        n_eval_episodes=config['training']['eval_episodes'],
        eval_freq=config['training']['eval_freq'] // n_envs,
        log_path=str(log_dir / "evaluations"),
        best_model_save_path=str(log_dir / "best_model"),
        deterministic=config['deterministic_eval'],
        as_agent=as_agent,
        as_results=as_results
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'] // n_envs,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_checkpoint"
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    print("✓ Callbacks ready")
    print()
    
    # Train
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"Evaluation frequency: {config['training']['eval_freq']:,} steps")
    print(f"Target: Beat AS baseline of ${as_results['mean_pnl']:.2f}")
    print()
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callback_list,
            log_interval=config['training']['log_interval'],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    training_time = time.time() - start_time
    
    print()
    print("="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Training time: {training_time/60:.1f} minutes")
    print()
    
    # Final evaluation
    print("Running final evaluation...")
    final_env = MarketMakerEnv()
    
    episode_pnls = []
    for ep in range(20):
        obs, info = final_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = final_env.step(action)
            done = terminated or truncated
        episode_pnls.append(info['total_pnl'])
    
    final_pnl = np.mean(episode_pnls)
    final_std = np.std(episode_pnls)
    final_sharpe = final_pnl / (final_std + 1e-6)
    
    print()
    print("="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"RL Agent (PPO):")
    print(f"  Mean PnL: ${final_pnl:.2f} ± ${final_std:.2f}")
    print(f"  Sharpe: {final_sharpe:.3f}")
    print()
    print(f"AS Baseline:")
    print(f"  Mean PnL: ${as_results['mean_pnl']:.2f} ± ${as_results['std_pnl']:.2f}")
    print(f"  Sharpe: {as_results['sharpe_ratio']:.3f}")
    print()
    
    improvement = final_pnl - as_results['mean_pnl']
    print(f"Improvement: ${improvement:.2f} ({improvement/abs(as_results['mean_pnl'])*100:.1f}%)")
    print()
    
    # Save final model
    final_model_path = log_dir / "final_model.zip"
    model.save(final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")
    print()
    
    print("="*60)
    print("To view training progress:")
    print(f"  tensorboard --logdir {log_dir}")
    print("="*60)


if __name__ == "__main__":
    import torch
    train_ppo()