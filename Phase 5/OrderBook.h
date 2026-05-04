#pragma once
#include <iostream>
#include <map>
#include <unordered_map>
#include <string>

// structure of single order
struct Order {
    std::string id;
    double price;
    int quantity;
    bool isBuy;
};

class OrderBook {
private:
    std::map<double, std::unordered_map<std::string, Order>> orderLevels;
    // O(1) lookup 
    std::unordered_map<std::string, Order> orderLookup;

public:
public:
    void addOrder(const std::string& id, double price, int quantity, bool isBuy);
    void modifyOrder(const std::string& id, double newPrice, int newQuantity);
    void deleteOrder(const std::string& id);

    bool hasOrder(const std::string& id) const;

    int getOrderCount() const {
        return static_cast<int>(orderLookup.size());
    }
};