"""
Avellaneda-Stoikov Market Making Model
Classic optimal market making strategy based on inventory risk
Reference: "High-frequency trading in a limit order book" (2008)
"""
import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from env.market_env import MarketMakerEnv


class AvellanedaStoikovAgent:
    """
    Implements the Avellaneda-Stoikov optimal market making strategy
    
    The model computes optimal bid/ask quotes based on:
    1. Risk aversion parameter (gamma)
    2. Current inventory (q)
    3. Time remaining (T - t)
    4. Market volatility (sigma)
    
    Key equations:
    - Reservation price: r = s - q * gamma * sigma^2 * (T - t)
    - Optimal spread: delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
    - Bid: r - delta/2
    - Ask: r + delta/2
    """
    
    def __init__(
        self,
        risk_aversion: float = 0.1,
        terminal_time: float = 1.0,
        order_arrival_rate: float = 1.0,
        min_spread: float = 0.0001,
        max_spread: float = 0.01
    ):
        """
        Args:
            risk_aversion: Risk aversion parameter (gamma). Higher = wider spreads
            terminal_time: Time horizon for inventory management
            order_arrival_rate: Rate parameter k for order arrivals
            min_spread: Minimum allowed spread (as fraction of price)
            max_spread: Maximum allowed spread (as fraction of price)
        """
        self.gamma = risk_aversion
        self.T = terminal_time
        self.k = order_arrival_rate
        self.min_spread = min_spread
        self.max_spread = max_spread
        
        # Track current state
        self.current_time = 0.0
        self.inventory = 0.0
        self.mid_price = 0.0
        self.volatility = 0.0
        
    def reset(self):
        """Reset agent state"""
        self.current_time = 0.0
        self.inventory = 0.0
        
    def get_quotes(
        self,
        mid_price: float,
        inventory: float,
        volatility: float,
        time_remaining: float
    ) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask prices
        
        Args:
            mid_price: Current mid price
            inventory: Current inventory position
            volatility: Market volatility estimate
            time_remaining: Time until end of trading (normalized 0-1)
        
        Returns:
            (bid_price, ask_price)
        """
        # Store state
        self.mid_price = mid_price
        self.inventory = inventory
        self.volatility = volatility
        
        # Time to maturity (scale by terminal time)
        tau = time_remaining * self.T
        tau = max(tau, 0.01)  # Avoid division by zero
        
        # Calculate reservation price
        # r = s - q * gamma * sigma^2 * tau
        reservation_price = mid_price - inventory * self.gamma * (volatility ** 2) * tau
        
        # Calculate optimal spread
        # delta = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/k)
        risk_term = self.gamma * (volatility ** 2) * tau
        
        # Asymptotic spread component (from order book liquidity)
        if self.k > 0:
            liquidity_term = (2.0 / self.gamma) * np.log(1 + self.gamma / self.k)
        else:
            liquidity_term = 0.0
        
        optimal_spread = risk_term + liquidity_term
        
        # Clip spread to reasonable bounds
        spread_fraction = optimal_spread / mid_price
        spread_fraction = np.clip(spread_fraction, self.min_spread, self.max_spread)
        spread_absolute = spread_fraction * mid_price
        
        # Calculate bid and ask
        bid_price = reservation_price - spread_absolute / 2
        ask_price = reservation_price + spread_absolute / 2
        
        # Ensure bid < mid < ask (sanity check)
        bid_price = min(bid_price, mid_price * 0.999)
        ask_price = max(ask_price, mid_price * 1.001)
        
        return bid_price, ask_price
    
    def get_action(
        self,
        observation: np.ndarray,
        env: Optional[MarketMakerEnv] = None
    ) -> np.ndarray:
        """
        Convert observation to action for gym environment
        
        Args:
            observation: Environment observation [inv, mid, spread, vol, time, ...]
            env: Environment instance (to get state info)
        
        Returns:
            action: [bid_offset, ask_offset] relative to mid price
        """
        # Parse observation
        norm_inv = observation[0]
        norm_price = observation[1]
        vol_estimate = observation[3]
        time_of_day = observation[4]
        
        # Denormalize
        if env is not None:
            inventory = norm_inv * env.max_inventory
            mid_price = env.initial_price * (1 + norm_price)
            volatility = vol_estimate
            time_remaining = 1.0 - time_of_day
        else:
            # Fallback if no env provided
            inventory = norm_inv * 10.0
            mid_price = 50000.0 * (1 + norm_price)
            volatility = vol_estimate
            time_remaining = 1.0 - time_of_day
        
        # Get optimal quotes
        bid_price, ask_price = self.get_quotes(
            mid_price=mid_price,
            inventory=inventory,
            volatility=volatility,
            time_remaining=time_remaining
        )
        
        # Convert to offsets (as fraction of mid)
        bid_offset = (bid_price - mid_price) / mid_price
        ask_offset = (ask_price - mid_price) / mid_price
        
        # Clip to action space bounds
        action = np.array([bid_offset, ask_offset], dtype=np.float32)
        action = np.clip(action, -0.005, 0.005)
        
        return action
    
    def __repr__(self) -> str:
        return (f"AvellanedaStoikovAgent(gamma={self.gamma}, T={self.T}, "
                f"k={self.k})")


def evaluate_agent(
    agent: AvellanedaStoikovAgent,
    env: MarketMakerEnv,
    n_episodes: int = 10,
    render: bool = False
) -> dict:
    """
    Evaluate AS agent performance
    
    Args:
        agent: AS agent instance
        env: Market environment
        n_episodes: Number of episodes to run
        render: Whether to print progress
    
    Returns:
        dict with performance metrics
    """
    episode_pnls = []
    episode_inventories = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset()
        
        episode_pnl = 0.0
        episode_inv = []
        done = False
        step = 0
        
        while not done:
            # Get action from AS model
            action = agent.get_action(obs, env)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_pnl = info['total_pnl']
            episode_inv.append(abs(info['inventory']))
            step += 1
            
            if render and step % 100 == 0:
                env.render()
        
        episode_pnls.append(episode_pnl)
        episode_inventories.append(np.mean(episode_inv))
        episode_lengths.append(step)
        
        if render:
            print(f"Episode {ep+1}/{n_episodes}: PnL=${episode_pnl:.2f}, "
                  f"Avg Inv={episode_inventories[-1]:.3f}")
    
    results = {
        'mean_pnl': np.mean(episode_pnls),
        'std_pnl': np.std(episode_pnls),
        'mean_inventory': np.mean(episode_inventories),
        'sharpe_ratio': np.mean(episode_pnls) / (np.std(episode_pnls) + 1e-6),
        'episode_pnls': episode_pnls,
        'episode_lengths': episode_lengths
    }
    
    return results


# Testing and comparison
if __name__ == "__main__":
    print("=" * 60)
    print("AVELLANEDA-STOIKOV BASELINE EVALUATION")
    print("=" * 60)
    print()
    
    # Create environment
    env = MarketMakerEnv()
    
    # Test different risk aversion levels
    gammas = [0.01, 0.1, 1.0]
    
    print("Testing different risk aversion parameters:")
    print()
    
    for gamma in gammas:
        print(f"Testing gamma={gamma}...")
        agent = AvellanedaStoikovAgent(risk_aversion=gamma)
        
        results = evaluate_agent(agent, env, n_episodes=5, render=False)
        
        print(f"  Mean PnL: ${results['mean_pnl']:.2f} ± ${results['std_pnl']:.2f}")
        print(f"  Sharpe: {results['sharpe_ratio']:.3f}")
        print(f"  Avg Inventory: {results['mean_inventory']:.3f}")
        print()
    
    # Detailed run with best gamma
    print("=" * 60)
    print("DETAILED RUN (gamma=0.1)")
    print("=" * 60)
    print()
    
    agent = AvellanedaStoikovAgent(risk_aversion=0.1)
    results = evaluate_agent(agent, env, n_episodes=3, render=True)
    
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Mean PnL: ${results['mean_pnl']:.2f}")
    print(f"Std PnL: ${results['std_pnl']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Mean Inventory: {results['mean_inventory']:.3f}")
    print()
    print("✓ AS Baseline ready! This is what your RL agent needs to beat.")