#include "order_manager.h"

int OrderManager::place_order(Side side, double price, int qty) {
    int id = next_order_id++;
    
    // create and initialize order
    auto order = std::make_unique<MyOrder>();
    order->id = id;
    order->side = side;
    order->price = price;
    order->quantity = qty;
    
    // store order and transfer ownership
    orders[id] = std::move(order);
    return id;
}

void OrderManager::cancel(int id) {
    // erase from map to trigger auto-cleanup
    orders.erase(id); 
}

void OrderManager::handle_fill(int id, int filled_qty) {
    auto it = orders.find(id);
    if (it == orders.end()) return;

    MyOrder* order = it->second.get();
    
    // update filled quantity
    order->filled += filled_qty;

    // check if fully filled
    if (order->filled >= order->quantity) {
        order->status = OrderStatus::Filled;
        orders.erase(it); // auto cleanup 
    } else {
        order->status = OrderStatus::PartiallyFilled;
    }
}

void OrderManager::print_active_orders() const {
    // print all active orders in memory
    for (const auto& pair : orders) {
        std::cout << "Order ID: " << pair.second->id 
                  << " Qty: " << pair.second->quantity 
                  << " Filled: " << pair.second->filled << "\n";
    }
}