#include "../include/OrderBook.hpp"

template <typename PriceType, typename OrderIdType>
void OrderBook<PriceType, OrderIdType>::add_order(OrderIdType id, const std::string& symbol, PriceType price, int quantity, bool is_buy) {
    // Acquire pre-allocated memory from the pool
    OrderPtr new_order = order_pool.acquire();
    
    // Initialize the order
    new_order->id = id;
    new_order->symbol = symbol;
    new_order->price = price;
    new_order->quantity = quantity;
    new_order->is_buy = is_buy;

    // Insert into the correct side of book
    if (is_buy) {
        bids.insert({price, std::move(new_order)});
    } else {
        asks.insert({price, std::move(new_order)});
    }
}

template <typename PriceType, typename OrderIdType>
void OrderBook<PriceType, OrderIdType>::print_book() const {
    std::cout << "\n=== Order Book State ===\n";
    std::cout << "[ASKS] (Sellers)\n";
    // Iterate asks in reverse so the lowest ask is at the bottom (closest to spread)
    for (auto it = asks.rbegin(); it != asks.rend(); ++it) {
        std::cout << "  Price: " << it->first << " | Qty: " << it->second->quantity << "\n";
    }
    
    std::cout << "------------------------\n";
    
    std::cout << "[BIDS] (Buyers)\n";
    for (const auto& [price, order] : bids) {
        std::cout << "  Price: " << price << " | Qty: " << order->quantity << "\n";
    }
    std::cout << "========================\n\n";
}

// Explicit instantiations for types we expect to use in main.cpp
template class OrderBook<double, int>;