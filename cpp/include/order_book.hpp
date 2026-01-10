#pragma once

#include "types.hpp"
#include <map>
#include <unordered_map>
#include <memory>
#include <vector>

namespace market_maker {

// Price level in the order book
class PriceLevel {
public:
    Price price;
    std::map<OrderID, Order> orders;  // Orders at this price
    Quantity total_quantity;
    
    PriceLevel(Price p) : price(p), total_quantity(0.0) {}
    
    void add_order(const Order& order);
    void remove_order(OrderID order_id);
    bool is_empty() const { return orders.empty(); }
};

// High-performance order book
class OrderBook {
private:
    // Bids: higher price = better, use reverse map
    std::map<Price, std::unique_ptr<PriceLevel>, std::greater<Price>> bids_;
    
    // Asks: lower price = better, use normal map
    std::map<Price, std::unique_ptr<PriceLevel>> asks_;
    
    // Quick order lookup
    std::unordered_map<OrderID, Order> order_map_;
    
    // Best prices (cached for O(1) access)
    Price best_bid_;
    Price best_ask_;
    
    // Statistics
    Timestamp last_update_time_;
    uint64_t total_orders_;
    
public:
    OrderBook();
    
    // Order management
    void add_order(const Order& order);
    void cancel_order(OrderID order_id);
    std::vector<Trade> match_order(const Order& order);
    
    // Queries
    Price get_best_bid() const { return best_bid_; }
    Price get_best_ask() const { return best_ask_; }
    double get_mid_price() const;
    double get_spread() const;
    
    // Get order book depth
    OrderBookSnapshot get_snapshot(int depth = 10) const;
    
    // Get quantity at price level
    Quantity get_bid_quantity(Price price) const;
    Quantity get_ask_quantity(Price price) const;
    
    // Statistics
    size_t get_total_orders() const { return order_map_.size(); }
    bool is_empty() const { return order_map_.empty(); }
    
    // Clear all orders
    void clear();
    
private:
    void update_best_prices();
    PriceLevel* get_or_create_level(Price price, Side side);
    void remove_empty_level(Price price, Side side);
};

} // namespace market_maker