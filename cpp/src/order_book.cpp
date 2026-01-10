#include "../include/order_book.hpp"
#include <algorithm>
#include <stdexcept>

namespace market_maker {

// ============================================
// PriceLevel Implementation
// ============================================

void PriceLevel::add_order(const Order& order) {
    orders[order.id] = order;
    total_quantity += order.quantity;
}

void PriceLevel::remove_order(OrderID order_id) {
    auto it = orders.find(order_id);
    if (it != orders.end()) {
        total_quantity -= it->second.quantity;
        orders.erase(it);
    }
}

// ============================================
// OrderBook Implementation
// ============================================

OrderBook::OrderBook() 
    : best_bid_(0), best_ask_(0), last_update_time_(0), total_orders_(0) {}

void OrderBook::add_order(const Order& order) {
    // Store in order map
    order_map_[order.id] = order;
    
    // Get or create price level
    PriceLevel* level = get_or_create_level(order.price, order.side);
    level->add_order(order);
    
    // Update best prices
    update_best_prices();
    
    last_update_time_ = order.timestamp;
    total_orders_++;
}

void OrderBook::cancel_order(OrderID order_id) {
    auto it = order_map_.find(order_id);
    if (it == order_map_.end()) {
        return;  // Order not found
    }
    
    const Order& order = it->second;
    Price price = order.price;
    Side side = order.side;
    
    // Remove from price level
    PriceLevel* level = nullptr;
    if (side == Side::BUY) {
        auto level_it = bids_.find(price);
        if (level_it != bids_.end()) {
            level = level_it->second.get();
        }
    } else {
        auto level_it = asks_.find(price);
        if (level_it != asks_.end()) {
            level = level_it->second.get();
        }
    }
    
    if (level) {
        level->remove_order(order_id);
        if (level->is_empty()) {
            remove_empty_level(price, side);
        }
    }
    
    // Remove from order map
    order_map_.erase(it);
    
    // Update best prices
    update_best_prices();
}

std::vector<Trade> OrderBook::match_order(const Order& order) {
    std::vector<Trade> trades;
    
    Quantity remaining_qty = order.quantity;
    
    // Match against opposite side
    if (order.side == Side::BUY) {
        // Match against asks
        while (remaining_qty > 0 && !asks_.empty()) {
            auto& best_ask_level = asks_.begin()->second;
            
            if (order.price < best_ask_level->price) {
                break;  // No more matches possible
            }
            
            // Match orders at this level
            auto order_it = best_ask_level->orders.begin();
            while (order_it != best_ask_level->orders.end() && remaining_qty > 0) {
                Order& maker_order = order_it->second;
                Quantity match_qty = std::min(remaining_qty, maker_order.quantity);
                
                // Create trade
                trades.emplace_back(
                    maker_order.id,
                    order.id,
                    maker_order.price,
                    match_qty,
                    order.timestamp,
                    Side::BUY
                );
                
                // Update quantities
                remaining_qty -= match_qty;
                maker_order.quantity -= match_qty;
                best_ask_level->total_quantity -= match_qty;
                
                // Remove filled orders
                if (maker_order.quantity <= 0) {
                    order_map_.erase(maker_order.id);
                    order_it = best_ask_level->orders.erase(order_it);
                } else {
                    ++order_it;
                }
            }
            
            // Remove empty level
            if (best_ask_level->is_empty()) {
                asks_.erase(asks_.begin());
            }
        }
    } else {
        // Match against bids
        while (remaining_qty > 0 && !bids_.empty()) {
            auto& best_bid_level = bids_.begin()->second;
            
            if (order.price > best_bid_level->price) {
                break;  // No more matches possible
            }
            
            // Match orders at this level
            auto order_it = best_bid_level->orders.begin();
            while (order_it != best_bid_level->orders.end() && remaining_qty > 0) {
                Order& maker_order = order_it->second;
                Quantity match_qty = std::min(remaining_qty, maker_order.quantity);
                
                // Create trade
                trades.emplace_back(
                    maker_order.id,
                    order.id,
                    maker_order.price,
                    match_qty,
                    order.timestamp,
                    Side::SELL
                );
                
                // Update quantities
                remaining_qty -= match_qty;
                maker_order.quantity -= match_qty;
                best_bid_level->total_quantity -= match_qty;
                
                // Remove filled orders
                if (maker_order.quantity <= 0) {
                    order_map_.erase(maker_order.id);
                    order_it = best_bid_level->orders.erase(order_it);
                } else {
                    ++order_it;
                }
            }
            
            // Remove empty level
            if (best_bid_level->is_empty()) {
                bids_.erase(bids_.begin());
            }
        }
    }
    
    // Add remaining quantity as new order
    if (remaining_qty > 0) {
        Order remaining_order = order;
        remaining_order.quantity = remaining_qty;
        add_order(remaining_order);
    }
    
    update_best_prices();
    
    return trades;
}

double OrderBook::get_mid_price() const {
    if (best_bid_ > 0 && best_ask_ > 0) {
        return (best_bid_ + best_ask_) / 2.0;
    }
    return 0.0;
}

double OrderBook::get_spread() const {
    if (best_bid_ > 0 && best_ask_ > 0) {
        return best_ask_ - best_bid_;
    }
    return 0.0;
}

OrderBookSnapshot OrderBook::get_snapshot(int depth) const {
    OrderBookSnapshot snapshot;
    snapshot.timestamp = last_update_time_;
    snapshot.best_bid = best_bid_;
    snapshot.best_ask = best_ask_;
    
    // Get bid levels
    int count = 0;
    for (const auto& [price, level] : bids_) {
        if (count >= depth) break;
        snapshot.bids.emplace_back(
            price, 
            level->total_quantity,
            static_cast<int>(level->orders.size())
        );
        count++;
    }
    
    // Get ask levels
    count = 0;
    for (const auto& [price, level] : asks_) {
        if (count >= depth) break;
        snapshot.asks.emplace_back(
            price,
            level->total_quantity,
            static_cast<int>(level->orders.size())
        );
        count++;
    }
    
    return snapshot;
}

Quantity OrderBook::get_bid_quantity(Price price) const {
    auto it = bids_.find(price);
    if (it != bids_.end()) {
        return it->second->total_quantity;
    }
    return 0.0;
}

Quantity OrderBook::get_ask_quantity(Price price) const {
    auto it = asks_.find(price);
    if (it != asks_.end()) {
        return it->second->total_quantity;
    }
    return 0.0;
}

void OrderBook::clear() {
    bids_.clear();
    asks_.clear();
    order_map_.clear();
    best_bid_ = 0;
    best_ask_ = 0;
}

void OrderBook::update_best_prices() {
    best_bid_ = bids_.empty() ? 0 : bids_.begin()->first;
    best_ask_ = asks_.empty() ? 0 : asks_.begin()->first;
}

PriceLevel* OrderBook::get_or_create_level(Price price, Side side) {
    if (side == Side::BUY) {
        auto it = bids_.find(price);
        if (it == bids_.end()) {
            it = bids_.emplace(price, std::make_unique<PriceLevel>(price)).first;
        }
        return it->second.get();
    } else {
        auto it = asks_.find(price);
        if (it == asks_.end()) {
            it = asks_.emplace(price, std::make_unique<PriceLevel>(price)).first;
        }
        return it->second.get();
    }
}

void OrderBook::remove_empty_level(Price price, Side side) {
    if (side == Side::BUY) {
        bids_.erase(price);
    } else {
        asks_.erase(price);
    }
}

} // namespace market_maker