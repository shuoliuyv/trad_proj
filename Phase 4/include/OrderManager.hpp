#pragma once

#include "Order.hpp"
#include <unordered_map>
#include <memory>
#include <type_traits>

// Enum class to type the order states
enum class OrderState {
    NEW,
    PARTIALLY_FILLED,
    FILLED,
    CANCELED
};

template <typename PriceType, typename OrderIdType>
class OrderManager {
    // restrict allowed types
    static_assert(std::is_integral<OrderIdType>::value, "Order ID must be an integer");

public:
    using OrderType = Order<PriceType, OrderIdType>;
    using OrderPtr = std::shared_ptr<OrderType>;

    OrderManager() = default;

    // Creates an order using RAII and registers it in the system
    OrderPtr create_order(OrderIdType id, const std::string& symbol, PriceType price, int quantity, bool is_buy);

    // updates the state of an existing order
    void update_order_state(OrderIdType id, OrderState state);

    // Retrieves an active order by ID
    OrderPtr get_order(OrderIdType id) const;

private:
    // lookups of orders 
    std::unordered_map<OrderIdType, OrderPtr> active_orders;
    std::unordered_map<OrderIdType, OrderState> order_states;
};