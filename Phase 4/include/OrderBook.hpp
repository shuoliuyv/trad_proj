#pragma once

#include "Order.hpp"
#include <map>
#include <vector>
#include <memory>
#include <iostream>

// Object pool to eliminate runtime new/delete overhead
template <typename T>
class MemoryPool {
private:
    std::vector<std::unique_ptr<T>> pool;
public:
    MemoryPool(size_t initial_size = 10000) {
        pool.reserve(initial_size);
        for(size_t i = 0; i < initial_size; ++i) {
            pool.push_back(std::make_unique<T>());
        }
    }
    
    std::unique_ptr<T> acquire() {
        if (pool.empty()) {
            return std::make_unique<T>();
        }
        auto obj = std::move(pool.back());
        pool.pop_back();
        return obj;
    }

    void release(std::unique_ptr<T> obj) {
        pool.push_back(std::move(obj));
    }
};

template <typename PriceType, typename OrderIdType>
class OrderBook {
    template <typename P, typename O> friend class MatchingEngine; //for MatchingEngine
public:
    using OrderType = Order<PriceType, OrderIdType>;
    using OrderPtr = std::unique_ptr<OrderType>;

    OrderBook() = default;

    // add a new order to the book
    void add_order(OrderIdType id, const std::string& symbol, PriceType price, int quantity, bool is_buy);
    
    // prints the current state of the L2 Order Book
    void print_book() const;

private:
    // memory pool for this order book
    MemoryPool<OrderType> order_pool;

    // Bids descending (highest price first)
    std::multimap<PriceType, OrderPtr, std::greater<PriceType>> bids;
    
    // Asks ascending (lowest price first) 
    std::multimap<PriceType, OrderPtr, std::less<PriceType>> asks;
};