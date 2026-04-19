#include "market_snapshot.h"

// update the bids order book
void MarketSnapshot::update_bid(double price, int qty) {
    if (qty == 0) {
        // std::map::erase  automatically destroy unique_ptr, preventing memory leaks
        bids.erase(price); 
    } else {
        // new unique_ptr if the price doesn't exist
        // or safely overwrites the existing unique_ptr
        bids[price] = std::make_unique<PriceLevel>(price, qty);
    }
}

// update ask
void MarketSnapshot::update_ask(double price, int qty) {
    if (qty == 0) {
        asks.erase(price);
    } else {
        asks[price] = std::make_unique<PriceLevel>(price, qty);
    }
}

// get the current best bid 
const PriceLevel* MarketSnapshot::get_best_bid() const {
    if (bids.empty()) {
        return nullptr; 
    }
    return bids.begin()->second.get(); 
}

// current best ask 
const PriceLevel* MarketSnapshot::get_best_ask() const {
    if (asks.empty()) {
        return nullptr; 
    }
    return asks.begin()->second.get();
}