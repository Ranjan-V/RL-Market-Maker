"""
Trading Logger - Records all trading activity
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd


class TradingLogger:
    """Log trading activity for analysis"""
    
    def __init__(self, symbol: str = "BTCUSDT", model_type: str = "SAC"):
        self.symbol = symbol
        self.model_type = model_type
        self.start_time = datetime.now()
        
        # Data storage
        self.steps = []
        
        # Create log directory
        self.log_dir = Path("data/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_step(
        self,
        step: int,
        market_state: Dict,
        inventory: float,
        pnl: float,
        orders: List
    ):
        """Log a single trading step"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'mid_price': market_state['mid_price'],
            'best_bid': market_state['best_bid'],
            'best_ask': market_state['best_ask'],
            'spread_bps': market_state['spread_bps'],
            'inventory': inventory,
            'pnl': pnl,
            'orders': [
                {
                    'side': side,
                    'price': price,
                    'quantity': qty
                }
                for side, price, qty in orders
            ]
        }
        
        self.steps.append(log_entry)
    
    def save(self) -> str:
        """Save log to file"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_type}_{self.symbol}_{timestamp}.json"
        filepath = self.log_dir / filename
        
        # Save JSON
        with open(filepath, 'w') as f:
            json.dump({
                'metadata': {
                    'symbol': self.symbol,
                    'model_type': self.model_type,
                    'start_time': self.start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_steps': len(self.steps)
                },
                'steps': self.steps
            }, f, indent=2)
        
        # Also save as CSV for easy analysis
        csv_file = filepath.with_suffix('.csv')
        df = pd.DataFrame(self.steps)
        df.to_csv(csv_file, index=False)
        
        return str(filepath)
    
    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics"""
        if not self.steps:
            return {}
        
        df = pd.DataFrame(self.steps)
        
        return {
            'total_steps': len(self.steps),
            'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'final_pnl': df['pnl'].iloc[-1],
            'max_pnl': df['pnl'].max(),
            'min_pnl': df['pnl'].min(),
            'max_inventory': df['inventory'].abs().max(),
            'avg_spread_bps': df['spread_bps'].mean(),
            'total_orders': sum(len(step['orders']) for step in self.steps)
        }