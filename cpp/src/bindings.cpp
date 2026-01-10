#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/types.hpp"
#include "../include/order_book.hpp"

namespace py = pybind11;
using namespace market_maker;

PYBIND11_MODULE(fast_orderbook, m) {
    m.doc() = "Fast C++ order book for market making";
    
    // Enums
    py::enum_<Side>(m, "Side")
        .value("BUY", Side::BUY)
        .value("SELL", Side::SELL)
        .export_values();
    
    // Order
    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def(py::init<OrderID, Price, Quantity, Side, Timestamp>())
        .def_readwrite("id", &Order::id)
        .def_readwrite("price", &Order::price)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("side", &Order::side)
        .def_readwrite("timestamp", &Order::timestamp);
    
    // Trade
    py::class_<Trade>(m, "Trade")
        .def(py::init<OrderID, OrderID, Price, Quantity, Timestamp, Side>())
        .def_readonly("maker_id", &Trade::maker_id)
        .def_readonly("taker_id", &Trade::taker_id)
        .def_readonly("price", &Trade::price)
        .def_readonly("quantity", &Trade::quantity)
        .def_readonly("timestamp", &Trade::timestamp)
        .def_readonly("aggressor_side", &Trade::aggressor_side);
    
    // Level2
    py::class_<Level2>(m, "Level2")
        .def(py::init<>())
        .def(py::init<Price, Quantity, int>())
        .def_readonly("price", &Level2::price)
        .def_readonly("total_quantity", &Level2::total_quantity)
        .def_readonly("order_count", &Level2::order_count);
    
    // OrderBookSnapshot
    py::class_<OrderBookSnapshot>(m, "OrderBookSnapshot")
        .def(py::init<>())
        .def_readonly("timestamp", &OrderBookSnapshot::timestamp)
        .def_readonly("bids", &OrderBookSnapshot::bids)
        .def_readonly("asks", &OrderBookSnapshot::asks)
        .def_readonly("best_bid", &OrderBookSnapshot::best_bid)
        .def_readonly("best_ask", &OrderBookSnapshot::best_ask)
        .def("get_mid_price", &OrderBookSnapshot::get_mid_price)
        .def("get_spread", &OrderBookSnapshot::get_spread);
    
    // OrderBook
    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("add_order", &OrderBook::add_order)
        .def("cancel_order", &OrderBook::cancel_order)
        .def("match_order", &OrderBook::match_order)
        .def("get_best_bid", &OrderBook::get_best_bid)
        .def("get_best_ask", &OrderBook::get_best_ask)
        .def("get_mid_price", &OrderBook::get_mid_price)
        .def("get_spread", &OrderBook::get_spread)
        .def("get_snapshot", &OrderBook::get_snapshot, py::arg("depth") = 10)
        .def("get_bid_quantity", &OrderBook::get_bid_quantity)
        .def("get_ask_quantity", &OrderBook::get_ask_quantity)
        .def("get_total_orders", &OrderBook::get_total_orders)
        .def("is_empty", &OrderBook::is_empty)
        .def("clear", &OrderBook::clear);
}