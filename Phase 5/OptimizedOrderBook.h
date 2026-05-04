#pragma once
#include <iostream>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <atomic>

struct OptimizedOrder {
    std::string id;
    double price;
    int quantity;
    bool isBuy;
    bool isActive; 
};

// Custom string hash
struct FastStringHash {
    size_t operator()(const std::string& key) const {
        size_t hash = 2166136261u; 
        for (char c : key) {
            hash ^= static_cast<size_t>(c);
            hash *= 16777619u; 
        }
        return hash;
    }
};

class OptimizedOrderBook {
private:
    std::vector<OptimizedOrder> orderPool; 

    std::map<double, std::unordered_map<std::string, size_t, FastStringHash>> orderLevels;
    std::unordered_map<std::string, size_t, FastStringHash> orderLookup;

    std::atomic<int> orderCount{0};

public:
    OptimizedOrderBook(size_t expectedVolume = 1000000) {
        orderPool.reserve(expectedVolume);
        orderLookup.reserve(expectedVolume);
    }

    void addOrder(const std::string& id, double price, int quantity, bool isBuy);
    void modifyOrder(const std::string& id, double newPrice, int newQuantity);
    void deleteOrder(const std::string& id);
    void processOrdersBatch(const std::vector<OptimizedOrder>& orders); 

    bool hasOrder(const std::string& id) const;
    const OptimizedOrder* getOrderById(const std::string& id) const;

    int getOrderCount() const {
        return orderCount.load(std::memory_order_relaxed);
    }
};