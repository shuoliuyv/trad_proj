#include "../include/OrderManager.hpp"

template <typename PriceType, typename OrderIdType>
std::shared_ptr<Order<PriceType, OrderIdType>> OrderManager<PriceType, OrderIdType>::create_order(
    OrderIdType id, const std::string& symbol, PriceType price, int quantity, bool is_buy) {
    
    // for exception safety and single memory allocation
    auto new_order = std::make_shared<OrderType>(id, symbol, price, quantity, is_buy);
    
    active_orders[id] = new_order;
    order_states[id] = OrderState::NEW;
    
    return new_order;
}

template <typename PriceType, typename OrderIdType>
void OrderManager<PriceType, OrderIdType>::update_order_state(OrderIdType id, OrderState state) {
    auto it = order_states.find(id);
    if (it != order_states.end()) {
        it->second = state;
        
        // memory management
        if (state == OrderState::FILLED || state == OrderState::CANCELED) {
            active_orders.erase(id);
        }
    }
}

template <typename PriceType, typename OrderIdType>
std::shared_ptr<Order<PriceType, OrderIdType>> OrderManager<PriceType, OrderIdType>::get_order(OrderIdType id) const {
    auto it = active_orders.find(id);
    if (it != active_orders.end()) {
        return it->second;
    }
    return nullptr; // return empty shared_ptr if not found
}

// template instantiations
template class OrderManager<double, int>;