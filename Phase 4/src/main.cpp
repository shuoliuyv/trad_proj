#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>
#include "../include/MarketData.hpp"
#include "../include/Order.hpp"
#include "../include/OrderBook.hpp"
#include "../include/MatchingEngine.hpp"
#include "../include/OrderManager.hpp"
#include "../include/TradeLogger.hpp"

using PriceType = double;
using OrderIdType = int;

int main() {
    std::vector<long long> latencies;
    const int num_ticks = 10000;
    latencies.reserve(num_ticks);

    OrderBook<double, int> orderBook;
    MatchingEngine<double, int> engine(orderBook);
    OrderManager<double, int> orderManager;
    TradeLogger logger("trades.log", 1000);
    MarketDataSimulator mdSimulator;

    // Pre-populate order book with liquidity so matches can occur
    for (int i = 0; i < 5000; ++i) {
        orderBook.add_order(i + 100000, "AAPL", 140.0 + (i % 5), 100, true);
        orderBook.add_order(i + 200000, "AAPL", 160.0 - (i % 5), 100, false);
    }

    for (int i = 0; i < num_ticks; ++i) {
        // Simulated tick
        MarketData tick = mdSimulator.generate_tick("AAPL", 150.0);
        
        // Create an aggressive order to guarantee a match against resting liquidity
        bool is_buy = i % 2 == 0;
        double price = is_buy ? tick.ask_price + 20.0 : tick.bid_price - 20.0; 
        
        auto order = orderManager.create_order(i, tick.symbol, price, 50, is_buy);
        
        // Match logic
        auto trades = engine.match_order(order, tick.timestamp);

        if (order->quantity == 0) {
            orderManager.update_order_state(order->id, OrderState::FILLED);
        } else if (!trades.empty()) {
            orderManager.update_order_state(order->id, OrderState::PARTIALLY_FILLED);
        }
                
        for (const auto& trade : trades) {
            logger.log_trade(trade);
            // Collect the latency calculated directly inside your match_order function
            latencies.push_back(trade.latency_ns); 
        }
    }

    if (latencies.empty()) {
        std::cout << "No trades executed.\n";
        return 0;
    }

    // Analyze latency
    auto min = *std::min_element(latencies.begin(), latencies.end());
    auto max = *std::max_element(latencies.begin(), latencies.end());
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

    double variance = 0.0;
    for (auto l : latencies) variance += (l - mean) * (l - mean);
    double stddev = std::sqrt(variance / latencies.size());

    std::sort(latencies.begin(), latencies.end());
    size_t p99_index = static_cast<size_t>(latencies.size() * 0.99);
    if (p99_index >= latencies.size()) {
        p99_index = latencies.size() - 1;
    }

    long long p99 = latencies[p99_index];

    std::cout << "Tick-to-Trade Latency (nanoseconds):\n";
    std::cout << "Min: " << min << " | Max: " << max << " | Mean: " << mean 
              << " | StdDev: " << stddev << " | P99: " << p99 << '\n';

    return 0;
}