#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>

#include "../include/MarketData.hpp"
#include "../include/OrderBook.hpp"
#include "../include/MatchingEngine.hpp"
#include "../include/OrderManager.hpp"

int main() {
    const int num_ticks = 10000;
    std::vector<long long> latencies;
    latencies.reserve(num_ticks);

    OrderBook<double, int> orderBook;
    MatchingEngine<double, int> engine(orderBook);
    OrderManager<double, int> orderManager;
    MarketDataSimulator mdSimulator;

    // Add resting liquidity
    for (int i = 0; i < 5000; ++i) {
        orderBook.add_order(i + 100000, "AAPL", 140.0 + (i % 5), 100, true);
        orderBook.add_order(i + 200000, "AAPL", 160.0 - (i % 5), 100, false);
    }

    for (int i = 0; i < num_ticks; ++i) {
        MarketData tick = mdSimulator.generate_tick("AAPL", 150.0);

        bool is_buy = i % 2 == 0;
        double price = is_buy ? tick.ask_price + 20.0 : tick.bid_price - 20.0;

        auto order = orderManager.create_order(i, tick.symbol, price, 50, is_buy);
        auto trades = engine.match_order(order, tick.timestamp);

        if (order->quantity == 0) {
            orderManager.update_order_state(order->id, OrderState::FILLED);
        } else if (!trades.empty()) {
            orderManager.update_order_state(order->id, OrderState::PARTIALLY_FILLED);
        }

        for (const auto& trade : trades) {
            latencies.push_back(trade.latency_ns);
        }
    }

    if (latencies.empty()) {
        std::cout << "Latency test failed: no trades executed.\n";
        return 1;
    }

    auto min = *std::min_element(latencies.begin(), latencies.end());
    auto max = *std::max_element(latencies.begin(), latencies.end());
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

    double variance = 0.0;
    for (auto l : latencies) {
        variance += (l - mean) * (l - mean);
    }
    double stddev = std::sqrt(variance / latencies.size());

    std::sort(latencies.begin(), latencies.end());
    size_t p99_index = static_cast<size_t>(latencies.size() * 0.99);
    if (p99_index >= latencies.size()) {
        p99_index = latencies.size() - 1;
    }
    long long p99 = latencies[p99_index];

    std::cout << "Latency Test Results:\n";
    std::cout << "Trades: " << latencies.size() << "\n";
    std::cout << "Min(ns): " << min << "\n";
    std::cout << "Max(ns): " << max << "\n";
    std::cout << "Mean(ns): " << mean << "\n";
    std::cout << "StdDev(ns): " << stddev << "\n";
    std::cout << "P99(ns): " << p99 << "\n";

    return 0;
}