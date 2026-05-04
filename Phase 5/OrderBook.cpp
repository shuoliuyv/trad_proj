#include "OrderBook.h"

// add a new order
void OrderBook::addOrder(const std::string& id, double price, int quantity, bool isBuy) {
    Order order = {id, price, quantity, isBuy};
    orderLevels[price][id] = order;
    orderLookup[id] = order;
}

// Modify an existing order
void OrderBook::modifyOrder(const std::string& id, double newPrice, int newQuantity) {
    // Ensure order exists
    auto it = orderLookup.find(id);
    if (it != orderLookup.end()) {
        Order oldOrder = it->second;
        
        // Remove the old order from its current price level
        orderLevels[oldOrder.price].erase(id);
        
        // Optimization: Clean up the price level if its empty
        if (orderLevels[oldOrder.price].empty()) {
            orderLevels.erase(oldOrder.price);
        }

        // add the order with the updated parameters
        addOrder(id, newPrice, newQuantity, oldOrder.isBuy);
    }
}

// Delete
void OrderBook::deleteOrder(const std::string& id) {
    auto it = orderLookup.find(id);
    if (it != orderLookup.end()) {
        Order order = it->second;
        
        orderLevels[order.price].erase(id);
        
        // Optimization: Clean up the price
        if (orderLevels[order.price].empty()) {
            orderLevels.erase(order.price);
        }
        
        orderLookup.erase(it);
    }
}

// Check if an order exists
bool OrderBook::hasOrder(const std::string& id) const {
    return orderLookup.find(id) != orderLookup.end();
}