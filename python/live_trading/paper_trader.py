"""
Live Paper Trading with Trained RL Agent
Runs your SAC/PPO agent on Binance Testnet
"""
import sys
from pathlib import Path
import time
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import json

sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, SAC
from binance_connector import BinanceTestnetConnector
from logger import TradingLogger
import os
from dotenv import load_dotenv


class LivePaperTrader:
    """
    Live paper trading system
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "SAC",
        symbol: str = "BTCUSDT",
        base_quantity: float = 0.001,  # BTC
        update_interval: int = 5,  # seconds
        max_inventory: float = 0.01,  # BTC
        initial_price: float = 50000.0
    ):
        """
        Args:
            model_path: Path to trained model
            model_type: 'SAC' or 'PPO'
            symbol: Trading pair
            base_quantity: Base order size
            update_interval: Seconds between updates
            max_inventory: Max position size
            initial_price: Reference price for normalization
        """
        self.symbol = symbol
        self.base_quantity = base_quantity
        self.update_interval = update_interval
        self.max_inventory = max_inventory
        self.initial_price = initial_price
        
        # Load model
        print(f"Loading {model_type} model from {model_path}...")
        if model_type.upper() == "SAC":
            self.model = SAC.load(model_path)
        elif model_type.upper() == "PPO":
            self.model = PPO.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        print("✓ Model loaded")
        
        # Load API credentials
        load_dotenv()
        api_key = os.getenv('BINANCE_TESTNET_API_KEY')
        api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Missing Binance testnet credentials in .env file!")
        
        # Initialize connector
        self.connector = BinanceTestnetConnector(api_key, api_secret)
        
        # Initialize logger
        self.logger = TradingLogger(symbol=symbol, model_type=model_type)
        
        # Trading state
        self.inventory = 0.0
        self.cash = 0.0
        self.start_balance = 0.0
        self.current_orders = {}  # {side: order_id}
        
        # Market state tracking
        self.price_history = []
        self.volatility = 0.0002  # Initial estimate
        
        # Performance
        self.trades_count = 0
        self.total_pnl = 0.0
        
    def get_observation(self, market_state: Dict) -> np.ndarray:
        """
        Convert market state to model observation
        
        Format: [inventory, mid_price, spread, volatility, time_of_day, 
                 recent_pnl, order_imbalance, microprice]
        """
        # Normalized inventory
        norm_inv = self.inventory / self.max_inventory
        
        # Normalized price
        mid_price = market_state['mid_price']
        norm_price = (mid_price - self.initial_price) / self.initial_price
        
        # Spread
        spread = market_state['spread']
        
        # Update volatility estimate
        self.price_history.append(mid_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        
        if len(self.price_history) > 10:
            recent_prices = np.array(self.price_history[-11:])
            returns = np.diff(recent_prices) / recent_prices[:-1]
            self.volatility = np.std(returns)
        
        # Time of day (normalized)
        now = datetime.now()
        time_of_day = (now.hour * 3600 + now.minute * 60 + now.second) / 86400
        
        # Recent PnL (normalized)
        norm_pnl = self.total_pnl / (self.initial_price * self.max_inventory)
        
        # Order imbalance (placeholder - could calculate from order book)
        order_imbalance = 0.0
        
        # Microprice (simplified)
        microprice = norm_price
        
        obs = np.array([
            norm_inv,
            norm_price,
            spread,
            self.volatility,
            time_of_day,
            norm_pnl,
            order_imbalance,
            microprice
        ], dtype=np.float32)
        
        return obs
    
    def place_quotes(self, action: np.ndarray, market_state: Dict):
        """
        Place limit orders based on model action
        
        Args:
            action: [bid_offset, ask_offset] from model
            market_state: Current market data
        """
        mid_price = market_state['mid_price']
        
        # Calculate bid and ask prices from action
        bid_offset = float(action[0])
        ask_offset = float(action[1])
        
        # ADJUSTMENT: Scale down offsets for live trading to get more fills
        # Model was trained on simulated environment, needs tighter spreads for real market
        scaling_factor = 0.3  # Make spreads 70% tighter
        bid_offset *= scaling_factor
        ask_offset *= scaling_factor
        
        bid_price = mid_price * (1 + bid_offset)
        ask_price = mid_price * (1 + ask_offset)
        
        # Round to valid price format (2 decimals for BTCUSDT)
        bid_price = round(bid_price, 2)
        ask_price = round(ask_price, 2)
        
        # Ensure ask > bid
        if ask_price <= bid_price:
            ask_price = bid_price + 0.01
        
        # Cancel existing orders
        self.cancel_all_orders()
        
        # Place new orders if within inventory limits
        orders_placed = []
        
        # Place bid (buy) if we're not at max long position
        if self.inventory < self.max_inventory:
            try:
                bid_order = self.connector.place_limit_order(
                    symbol=self.symbol,
                    side='BUY',
                    quantity=self.base_quantity,
                    price=bid_price
                )
                if bid_order:
                    self.current_orders['BUY'] = bid_order['orderId']
                    orders_placed.append(('BUY', bid_price, self.base_quantity))
            except Exception as e:
                print(f"Failed to place bid: {e}")
        
        # Place ask (sell) if we're not at max short position
        if self.inventory > -self.max_inventory:
            try:
                ask_order = self.connector.place_limit_order(
                    symbol=self.symbol,
                    side='SELL',
                    quantity=self.base_quantity,
                    price=ask_price
                )
                if ask_order:
                    self.current_orders['SELL'] = ask_order['orderId']
                    orders_placed.append(('SELL', ask_price, self.base_quantity))
            except Exception as e:
                print(f"Failed to place ask: {e}")
        
        return orders_placed
    
    def cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            self.connector.cancel_all_orders(self.symbol)
            self.current_orders = {}
        except Exception as e:
            print(f"Failed to cancel orders: {e}")
    
    def update_fills(self):
        """Check for filled orders and update inventory"""
        try:
            open_orders = self.connector.get_open_orders(self.symbol)
            
            if open_orders is None:
                return
            
            open_order_ids = {order['orderId'] for order in open_orders}
            
            # Check if any of our orders were filled
            for side, order_id in list(self.current_orders.items()):
                if order_id not in open_order_ids:
                    # Order was filled!
                    if side == 'BUY':
                        self.inventory += self.base_quantity
                        self.trades_count += 1
                        print(f"  ✓ BUY filled: {self.base_quantity} BTC")
                    elif side == 'SELL':
                        self.inventory -= self.base_quantity
                        self.trades_count += 1
                        print(f"  ✓ SELL filled: {self.base_quantity} BTC")
                    
                    del self.current_orders[side]
        
        except Exception as e:
            print(f"Failed to update fills: {e}")
    
    def calculate_pnl(self, market_state: Dict) -> float:
        """Calculate mark-to-market PnL"""
        mid_price = market_state['mid_price']
        mtm_value = self.inventory * mid_price
        return self.cash + mtm_value - self.start_balance
    
    def run(self, duration_minutes: int = 60):
        """
        Run live paper trading
        
        Args:
            duration_minutes: How long to run (minutes)
        """
        print("=" * 60)
        print("LIVE PAPER TRADING")
        print("=" * 60)
        print()
        
        # Test connection
        if not self.connector.test_connection():
            print("❌ Failed to connect to Binance testnet")
            return
        
        print()
        
        # Get starting balance
        usdt_balance = self.connector.get_balance("USDT")
        btc_balance = self.connector.get_balance("BTC")
        
        print(f"Starting balances:")
        print(f"  USDT: ${usdt_balance:,.2f}" if usdt_balance else "  USDT: N/A")
        print(f"  BTC: {btc_balance:.6f}" if btc_balance else "  BTC: N/A")
        print()
        
        self.start_balance = usdt_balance if usdt_balance else 10000.0
        self.cash = self.start_balance
        
        print(f"Trading {self.symbol}")
        print(f"Update interval: {self.update_interval}s")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Base quantity: {self.base_quantity} BTC")
        print()
        print("=" * 60)
        print()
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        step = 0
        
        try:
            while time.time() < end_time:
                step += 1
                
                # Get market state
                market_state = self.connector.get_market_state(self.symbol)
                if not market_state:
                    print("Failed to get market state, retrying...")
                    time.sleep(self.update_interval)
                    continue
                
                # Check for fills
                self.update_fills()
                
                # Get observation
                obs = self.get_observation(market_state)
                
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Place quotes
                orders = self.place_quotes(action, market_state)
                
                # Calculate PnL
                self.total_pnl = self.calculate_pnl(market_state)
                
                # Log
                self.logger.log_step(
                    step=step,
                    market_state=market_state,
                    inventory=self.inventory,
                    pnl=self.total_pnl,
                    orders=orders
                )
                
                # Print status
                elapsed = (time.time() - start_time) / 60
                print(f"[{elapsed:5.1f}m] "
                      f"Price: ${market_state['mid_price']:8,.2f} | "
                      f"Inv: {self.inventory:+7.4f} | "
                      f"PnL: ${self.total_pnl:+8.2f} | "
                      f"Trades: {self.trades_count:3d} | "
                      f"Orders: {len(orders)}")
                
                # Wait
                time.sleep(self.update_interval)
        
        except KeyboardInterrupt:
            print("\n\nTrading interrupted by user")
        
        finally:
            # Cleanup
            print("\nCleaning up...")
            self.cancel_all_orders()
            
            # Final summary
            print()
            print("=" * 60)
            print("TRADING SUMMARY")
            print("=" * 60)
            print(f"Duration: {(time.time() - start_time) / 60:.1f} minutes")
            print(f"Total trades: {self.trades_count}")
            print(f"Final inventory: {self.inventory:+.6f} BTC")
            print(f"Final PnL: ${self.total_pnl:+.2f}")
            print()
            
            # Save log
            log_file = self.logger.save()
            print(f"✓ Log saved to: {log_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run live paper trading')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--type', type=str, default='SAC', choices=['SAC', 'PPO'], help='Model type')
    parser.add_argument('--duration', type=int, default=60, help='Trading duration (minutes)')
    parser.add_argument('--interval', type=int, default=5, help='Update interval (seconds)')
    
    args = parser.parse_args()
    
    trader = LivePaperTrader(
        model_path=args.model,
        model_type=args.type,
        update_interval=args.interval
    )
    
    trader.run(duration_minutes=args.duration)