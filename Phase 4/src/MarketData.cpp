#include "../include/MarketData.hpp"
#include <random>

MarketData MarketDataSimulator::generate_tick(const std::string& symbol, double base_price) {
    // static random number generator
    // fixed seed for reproducible benchmark results
    static std::mt19937 generator(1234); 
    static std::uniform_real_distribution<double> spread_dist(0.01, 0.05);
    static std::uniform_real_distribution<double> price_dist(-0.5, 0.5);

    double current_price = base_price + price_dist(generator);
    double spread = spread_dist(generator);

    MarketData tick;
    tick.symbol = symbol;
    tick.bid_price = current_price - (spread / 2);
    tick.ask_price = current_price + (spread / 2);
    
    tick.timestamp = std::chrono::high_resolution_clock::now();

    return tick;
}

std::vector<MarketData> MarketDataSimulator::generate_batch(int num_ticks) {
    std::vector<MarketData> batch;
    // ore-allocate memory 
    batch.reserve(num_ticks); 
    
    for (int i = 0; i < num_ticks; ++i) {
        batch.push_back(generate_tick("AAPL", 150.0));
    }
    
    return batch;
}