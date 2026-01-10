"""
Binance Testnet API Connector
Connects to Binance Spot Testnet for paper trading
"""
import time
import hmac
import hashlib
import requests
from typing import Dict, List, Optional
from datetime import datetime
import json


class BinanceTestnetConnector:
    """
    Connector for Binance Spot Testnet
    
    Get API keys from: https://testnet.binance.vision/
    """
    
    BASE_URL = "https://testnet.binance.vision"
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize connector
        
        Args:
            api_key: Binance testnet API key
            api_secret: Binance testnet API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': api_key
        })
        
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature"""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make API request"""
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, params=params, timeout=10)
            elif method == 'DELETE':
                response = self.session.delete(url, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return None
    
    # ============================================
    # Market Data
    # ============================================
    
    def get_ticker(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """Get current ticker price"""
        return self._request('GET', '/api/v3/ticker/price', {'symbol': symbol})
    
    def get_order_book(self, symbol: str = "BTCUSDT", limit: int = 10) -> Optional[Dict]:
        """Get order book depth"""
        return self._request('GET', '/api/v3/depth', {
            'symbol': symbol,
            'limit': limit
        })
    
    def get_recent_trades(self, symbol: str = "BTCUSDT", limit: int = 10) -> Optional[List]:
        """Get recent trades"""
        return self._request('GET', '/api/v3/trades', {
            'symbol': symbol,
            'limit': limit
        })
    
    # ============================================
    # Account
    # ============================================
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        return self._request('GET', '/api/v3/account', signed=True)
    
    def get_balance(self, asset: str = "USDT") -> Optional[float]:
        """Get balance for specific asset"""
        account = self.get_account_info()
        if account:
            for balance in account.get('balances', []):
                if balance['asset'] == asset:
                    return float(balance['free'])
        return None
    
    # ============================================
    # Orders
    # ============================================
    
    def place_limit_order(
        self, 
        symbol: str,
        side: str,  # 'BUY' or 'SELL'
        quantity: float,
        price: float
    ) -> Optional[Dict]:
        """Place limit order"""
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'LIMIT',
            'timeInForce': 'GTC',  # Good till cancelled
            'quantity': quantity,
            'price': price
        }
        return self._request('POST', '/api/v3/order', params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: int) -> Optional[Dict]:
        """Cancel an order"""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        return self._request('DELETE', '/api/v3/order', params, signed=True)
    
    def get_open_orders(self, symbol: str = "BTCUSDT") -> Optional[List]:
        """Get all open orders"""
        return self._request('GET', '/api/v3/openOrders', {'symbol': symbol}, signed=True)
    
    def cancel_all_orders(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """Cancel all open orders for symbol"""
        return self._request('DELETE', '/api/v3/openOrders', {'symbol': symbol}, signed=True)
    
    # ============================================
    # Helper Methods
    # ============================================
    
    def get_market_state(self, symbol: str = "BTCUSDT") -> Dict:
        """Get comprehensive market state"""
        ticker = self.get_ticker(symbol)
        order_book = self.get_order_book(symbol, limit=5)
        
        if not ticker or not order_book:
            return None
        
        # Calculate mid price
        best_bid = float(order_book['bids'][0][0]) if order_book['bids'] else 0
        best_ask = float(order_book['asks'][0][0]) if order_book['asks'] else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else float(ticker['price'])
        
        # Calculate spread
        spread = (best_ask - best_bid) / mid_price if best_bid and best_ask else 0
        
        return {
            'symbol': symbol,
            'mid_price': mid_price,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_bps': spread * 10000,  # Basis points
            'timestamp': datetime.now().isoformat()
        }
    
    def test_connection(self) -> bool:
        """Test if connection works"""
        try:
            ticker = self.get_ticker("BTCUSDT")
            account = self.get_account_info()
            
            if ticker and account:
                print("✓ Connection successful!")
                print(f"  BTC Price: ${float(ticker['price']):,.2f}")
                print(f"  Account status: {account.get('accountType', 'N/A')}")
                return True
            else:
                print("✗ Connection failed")
                return False
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False


# Test script
if __name__ == "__main__":
    print("=" * 60)
    print("BINANCE TESTNET CONNECTOR TEST")
    print("=" * 60)
    print()
    print("⚠️  You need to set up API keys first!")
    print()
    print("Steps:")
    print("1. Go to: https://testnet.binance.vision/")
    print("2. Click 'Generate HMAC_SHA256 Key'")
    print("3. Save API Key and Secret Key")
    print("4. Create .env file with:")
    print("   BINANCE_TESTNET_API_KEY=your_key_here")
    print("   BINANCE_TESTNET_API_SECRET=your_secret_here")
    print()
    
    # Try loading from environment
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    api_key = os.getenv('BINANCE_TESTNET_API_KEY')
    api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')
    
    if api_key and api_secret:
        print("Found API credentials in .env file")
        print()
        
        connector = BinanceTestnetConnector(api_key, api_secret)
        
        print("Testing connection...")
        if connector.test_connection():
            print()
            print("Getting market state...")
            state = connector.get_market_state("BTCUSDT")
            if state:
                print(f"  Mid Price: ${state['mid_price']:,.2f}")
                print(f"  Spread: {state['spread_bps']:.2f} bps")
                print(f"  Best Bid: ${state['best_bid']:,.2f}")
                print(f"  Best Ask: ${state['best_ask']:,.2f}")
    else:
        print("❌ No API credentials found!")
        print("Please create .env file with your Binance testnet keys")