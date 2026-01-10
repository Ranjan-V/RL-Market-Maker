#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace market_maker {

// Price represented as integer (in ticks)
using Price = int64_t;
using Quantity = double;
using Timestamp = int64_t;
using OrderID = uint64_t;

// Side of the order
enum class Side : uint8_t {
    BUY = 0,
    SELL = 1
};

// Order structure
struct Order {
    OrderID id;
    Price price;
    Quantity quantity;
    Side side;
    Timestamp timestamp;
    
    Order() : id(0), price(0), quantity(0.0), side(Side::BUY), timestamp(0) {}
    
    Order(OrderID id_, Price price_, Quantity qty_, Side side_, Timestamp ts_)
        : id(id_), price(price_), quantity(qty_), side(side_), timestamp(ts_) {}
};

// Trade structure
struct Trade {
    OrderID maker_id;
    OrderID taker_id;
    Price price;
    Quantity quantity;
    Timestamp timestamp;
    Side aggressor_side;
    
    Trade(OrderID maker, OrderID taker, Price p, Quantity q, Timestamp ts, Side side)
        : maker_id(maker), taker_id(taker), price(p), quantity(q), 
          timestamp(ts), aggressor_side(side) {}
};

// Level 2 snapshot (order book level)
struct Level2 {
    Price price;
    Quantity total_quantity;
    int order_count;
    
    Level2() : price(0), total_quantity(0.0), order_count(0) {}
    Level2(Price p, Quantity q, int count) 
        : price(p), total_quantity(q), order_count(count) {}
};

// Full order book snapshot
struct OrderBookSnapshot {
    Timestamp timestamp;
    std::vector<Level2> bids;  // Sorted descending by price
    std::vector<Level2> asks;  // Sorted ascending by price
    Price best_bid;
    Price best_ask;
    
    OrderBookSnapshot() : timestamp(0), best_bid(0), best_ask(0) {}
    
    double get_mid_price() const {
        if (best_bid > 0 && best_ask > 0) {
            return (best_bid + best_ask) / 2.0;
        }
        return 0.0;
    }
    
    double get_spread() const {
        if (best_bid > 0 && best_ask > 0) {
            return best_ask - best_bid;
        }
        return 0.0;
    }
};

// Market maker quote
struct Quote {
    Price bid_price;
    Price ask_price;
    Quantity bid_quantity;
    Quantity ask_quantity;
    Timestamp timestamp;
    
    Quote() : bid_price(0), ask_price(0), bid_quantity(0.0), 
              ask_quantity(0.0), timestamp(0) {}
    
    Quote(Price bid, Price ask, Quantity bid_qty, Quantity ask_qty, Timestamp ts)
        : bid_price(bid), ask_price(ask), bid_quantity(bid_qty),
          ask_quantity(ask_qty), timestamp(ts) {}
};

// Performance metrics
struct PerformanceMetrics {
    double total_pnl;
    double sharpe_ratio;
    double max_drawdown;
    double win_rate;
    int total_trades;
    Quantity avg_inventory;
    double processing_time_us;  // Microseconds
    
    PerformanceMetrics() 
        : total_pnl(0.0), sharpe_ratio(0.0), max_drawdown(0.0),
          win_rate(0.0), total_trades(0), avg_inventory(0.0),
          processing_time_us(0.0) {}
};

} // namespace market_maker