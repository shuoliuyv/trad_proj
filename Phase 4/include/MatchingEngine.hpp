#pragma once

#include "OrderBook.hpp"
#include <vector>
#include <string>
#include <chrono>
#include <memory>


struct Trade {
    int buyer_id;
    int seller_id;
    std::string symbol;
    double price;
    int quantity;
    long long latency_ns;
};

template <typename PriceType, typename OrderIdType>
class MatchingEngine {
public:
    using OrderPtr = std::shared_ptr<Order<PriceType, OrderIdType>>;

    // Pass OrderBook by reference 
    MatchingEngine(OrderBook<PriceType, OrderIdType>& book);

    // takes an incoming order and the timestamp of the market tick
    std::vector<Trade> match_order(OrderPtr incoming_order, std::chrono::high_resolution_clock::time_point tick_time);

private:
    OrderBook<PriceType, OrderIdType>& order_book;
};