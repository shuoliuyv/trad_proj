#pragma once

#include <map>
#include <memory>
#include <functional> 

struct PriceLevel {
    double price;
    int quantity;

    PriceLevel(double p, int q) : price(p), quantity(q) {}
};

class MarketSnapshot {
private:
    // instant access to the best prices- the top of book.
    std::map<double, std::unique_ptr<PriceLevel>, std::greater<double>> bids;
    std::map<double, std::unique_ptr<PriceLevel>> asks;

public:
    void update_bid(double price, int qty);
    void update_ask(double price, int qty);

    const PriceLevel* get_best_bid() const;
    const PriceLevel* get_best_ask() const;
};