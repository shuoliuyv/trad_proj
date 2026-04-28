#pragma once

#include <string>
#include <chrono>
#include <vector>

// Use alignas(64) to enforce cache-line alignment
struct alignas(64) MarketData {
    std::string symbol;
    double bid_price;
    double ask_price;
    std::chrono::high_resolution_clock::time_point timestamp;
};

class MarketDataSimulator {
public:
    MarketDataSimulator() = default;

    // generates single simulated market tick
    MarketData generate_tick(const std::string& symbol, double base_price);

    // simulates a batch of market data ticks 
    std::vector<MarketData> generate_batch(int num_ticks);
};