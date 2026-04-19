# Phase 3: Local Order Book & Core Trading Infrastructure

## Overview
This project implements a small real-time trading engine core that maintains market state and tracks order lifecycles. The system is designed with an emphasis on memory safety, clear ownership, and efficient execution.

## Architecture

- **MarketSnapshot**  
  Maintains real-time bid and ask levels using `std::map`. This allows ordered price storage and fast access to the best bid and best ask.

- **OrderManager**  
  Tracks submitted orders, updates fill status, and maintains a ledger of active orders throughout their lifecycle.

- **Strategy Loop**  
  Connects market data to trading actions. It processes a simulated feed, updates the local book, and triggers order decisions based on simple strategy logic.

## Memory Management & Safety

- **Zero Raw Pointers**  
  All dynamically managed objects, such as `PriceLevel` and `MyOrder`, are owned through `std::unique_ptr`.

- **RAII-Based Design**  
  Resource lifetime is tied to object scope and container ownership. When an object is erased from a container or the program exits, its memory is released automatically.

- **No Manual `new` / `delete`**  
  The system avoids direct manual memory management, reducing the risk of memory leaks, dangling pointers, and double frees.

- **Clear Ownership Model**  
  Each dynamically allocated object has a single owner, making object lifetime explicit and safer to reason about.

## How to Compile and Run

### Compilation
Use G++ with C++17 support:

```bash
g++ -std=c++17 main.cpp market_snapshot.cpp order_manager.cpp -o trad_sys