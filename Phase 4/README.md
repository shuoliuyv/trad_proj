# Phase 4: Build & Benchmark a High-Frequency Trading System in C++

## Design Overview

This project implements a small high-frequency trading system prototype in C++. The system simulates market data, manages client orders, matches orders through a limit order book, logs executed trades, and measures tick-to-trade latency.

The project is designed with an emphasis on modular structure, memory safety, clear ownership, and efficient execution. It uses smart pointers, RAII, a simple memory pool, cache-line alignment, and batched trade logging.

## Architecture
```text
+------------------+       +-----------------+       +------------------+
| MarketDataFeed   |  -->  | OrderBook       | <---> | OrderManagement  |
+------------------+       +-----------------+       +------------------+
                                 |     |
                                 v     v
                         +-------------------+      +------------------+
                         | MatchingEngine    | ---> | TradeLogger      |
                         +-------------------+      +------------------+
```
- **MarketDataSimulator**  
  Generates simulated market ticks with symbol, bid price, ask price, and timestamp. The `MarketData` struct uses `alignas(64)` to enforce cache-line alignment.

- **OrderBook**  
  Maintains resting buy and sell orders using `std::multimap`. Buy orders are sorted from highest to lowest price, while sell orders are sorted from lowest to highest price. Resting orders are owned through `std::unique_ptr`.

- **OrderManager**  
  Creates client orders and tracks their lifecycle states, including `NEW`, `PARTIALLY_FILLED`, `FILLED`, and `CANCELED`. It uses `std::shared_ptr` for order tracking and `static_assert` to make sure the order ID type is integral.

- **MatchingEngine**  
  Matches incoming buy orders against resting sell orders and incoming sell orders against resting buy orders. It supports partial fills, removes fully filled resting orders, and records tick-to-trade latency for each executed trade.

- **TradeLogger**  
  Stores executed trades in a buffer and writes them to `trades.log` in batches. This reduces frequent file I/O and keeps logging separated from the core matching logic.

- **Latency Test**  
  Runs a standalone benchmark using simulated ticks and preloaded liquidity. It reports latency statistics such as minimum, maximum, mean, standard deviation, and P99 latency.

## How to Compile and Run

### Option 1: Build with CMake

If CMake is installed, build the project from the `Phase 4` directory:

    mkdir -p build
    cd build
    cmake ..
    make

Run the main application:

    ./hft_app

Run the latency test:

    ./latency_test

### Option 2: Build Directly with G++

If CMake is not installed, the project can also be compiled directly with `g++`.

Compile the main application:

    g++ -std=c++17 -O3 \
    src/main.cpp src/MarketData.cpp src/OrderBook.cpp src/MatchingEngine.cpp src/OrderManager.cpp src/TradeLogger.cpp \
    -Iinclude -o hft_app

Run the main application:

    ./hft_app

Compile the latency test:

    g++ -std=c++17 -O3 \
    test/test_latency.cpp src/MarketData.cpp src/OrderBook.cpp src/MatchingEngine.cpp src/OrderManager.cpp \
    -Iinclude -o latency_test

Run the latency test:

    ./latency_test