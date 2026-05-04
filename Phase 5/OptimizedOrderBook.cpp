#include "OptimizedOrderBook.h"

void OptimizedOrderBook::addOrder(const std::string& id, double price, int quantity, bool isBuy) {
    size_t poolIndex = orderPool.size();
    orderPool.push_back({id, price, quantity, isBuy, true});

    orderLevels[price][id] = poolIndex;
    orderLookup[id] = poolIndex;

    orderCount.fetch_add(1, std::memory_order_relaxed);
}

void OptimizedOrderBook::modifyOrder(const std::string& id, double newPrice, int newQuantity) {
    auto it = orderLookup.find(id);

    if (it != orderLookup.end()) {
        size_t index = it->second;
        double oldPrice = orderPool[index].price;

        auto levelIt = orderLevels.find(oldPrice);
        if (levelIt != orderLevels.end()) {
            levelIt->second.erase(id);

            if (levelIt->second.empty()) {
                orderLevels.erase(levelIt);
            }
        }

        orderPool[index].price = newPrice;
        orderPool[index].quantity = newQuantity;

        orderLevels[newPrice][id] = index;
    } else {
        std::cerr << "Order " << id << " not found for modification.\n";
    }
}

void OptimizedOrderBook::deleteOrder(const std::string& id) {
    auto it = orderLookup.find(id);

    if (it != orderLookup.end()) {
        size_t index = it->second;
        double price = orderPool[index].price;

        auto levelIt = orderLevels.find(price);
        if (levelIt != orderLevels.end()) {
            levelIt->second.erase(id);

            if (levelIt->second.empty()) {
                orderLevels.erase(levelIt);
            }
        }

        orderPool[index].isActive = false;
        orderLookup.erase(it);

        orderCount.fetch_sub(1, std::memory_order_relaxed);
    }
}

void OptimizedOrderBook::processOrdersBatch(const std::vector<OptimizedOrder>& orders) { 
    size_t n = orders.size();
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        addOrder(orders[i].id, orders[i].price, orders[i].quantity, orders[i].isBuy);
        addOrder(orders[i + 1].id, orders[i + 1].price, orders[i + 1].quantity, orders[i + 1].isBuy);
        addOrder(orders[i + 2].id, orders[i + 2].price, orders[i + 2].quantity, orders[i + 2].isBuy);
        addOrder(orders[i + 3].id, orders[i + 3].price, orders[i + 3].quantity, orders[i + 3].isBuy);
    }

    for (; i < n; ++i) {
        addOrder(orders[i].id, orders[i].price, orders[i].quantity, orders[i].isBuy);
    }
}

bool OptimizedOrderBook::hasOrder(const std::string& id) const {
    auto it = orderLookup.find(id);
    if (it == orderLookup.end()) {
        return false;
    }

    size_t index = it->second;
    return orderPool[index].isActive;
}

const OptimizedOrder* OptimizedOrderBook::getOrderById(const std::string& id) const {
    auto it = orderLookup.find(id);
    if (it == orderLookup.end()) {
        return nullptr;
    }

    size_t index = it->second;
    if (!orderPool[index].isActive) {
        return nullptr;
    }

    return &orderPool[index];
}