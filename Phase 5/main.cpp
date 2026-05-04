#include "OrderBook.h"
#include "OptimizedOrderBook.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <cmath>

//step 4 generate order
std::vector<OptimizedOrder> generateOrders(int numOrders) {
    std::vector<OptimizedOrder> orders;
    orders.reserve(numOrders);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> priceDist(50.0, 100.0);
    std::uniform_int_distribution<int> quantityDist(1, 500);

    for (int i = 0; i < numOrders; ++i) {
        std::string id = "ORD" + std::to_string(i);
        double price = priceDist(rng);
        int quantity = quantityDist(rng);
        bool isBuy = (i % 2 == 0);

        orders.push_back({id, price, quantity, isBuy, true});
    }

    return orders;
}

//step 2
void profileBaseImplementation() {
    const int NUM_ORDERS = 100000;
    std::vector<OptimizedOrder> orders = generateOrders(NUM_ORDERS);
    OrderBook book;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto& order : orders) {
        book.addOrder(order.id, order.price, order.quantity, order.isBuy);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    int found = 0;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (const auto& order : orders) {
        if (book.hasOrder(order.id)) {
            found++;
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    auto t4 = std::chrono::high_resolution_clock::now();
    for (const auto& order : orders) {
        book.modifyOrder(order.id, order.price + 0.01, order.quantity + 1);
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    auto t6 = std::chrono::high_resolution_clock::now();
    for (const auto& order : orders) {
        book.deleteOrder(order.id);
    }
    auto t7 = std::chrono::high_resolution_clock::now();

    double addTime = std::chrono::duration<double>(t1 - t0).count();
    double lookupTime = std::chrono::duration<double>(t3 - t2).count();
    double modifyTime = std::chrono::duration<double>(t5 - t4).count();
    double deleteTime = std::chrono::duration<double>(t7 - t6).count();

    std::cout << "Base Profiling (" << NUM_ORDERS << " orders)\n";
    std::cout << "Add:    " << addTime << " s, "
              << addTime * 1e9 / NUM_ORDERS << " ns/order\n";
    std::cout << "Lookup: " << lookupTime << " s, "
              << lookupTime * 1e9 / NUM_ORDERS << " ns/order, found = "
              << found << "\n";
    std::cout << "Modify: " << modifyTime << " s, "
              << modifyTime * 1e9 / NUM_ORDERS << " ns/order\n";
    std::cout << "Delete: " << deleteTime << " s, "
              << deleteTime * 1e9 / NUM_ORDERS << " ns/order\n\n";
}


//step 4 test
void testAddOrder() {
    OptimizedOrderBook book;
    book.addOrder("ORD001", 50.10, 100, true);

    assert(book.getOrderCount() == 1);
    assert(book.hasOrder("ORD001"));
}

void testModifyOrder() {
    OptimizedOrderBook book;
    book.addOrder("ORD001", 50.10, 100, true);

    book.modifyOrder("ORD001", 51.25, 200);

    const OptimizedOrder* order = book.getOrderById("ORD001");

    assert(order != nullptr);
    assert(std::abs(order->price - 51.25) < 1e-9);
    assert(order->quantity == 200);
    assert(book.getOrderCount() == 1);
}

void testDeleteOrder() {
    OptimizedOrderBook book;
    book.addOrder("ORD001", 50.10, 100, true);

    book.deleteOrder("ORD001");

    assert(book.getOrderCount() == 0);
    assert(!book.hasOrder("ORD001"));
}

void runUnitTests() {
    std::cout << "Running unit tests...\n";

    testAddOrder();
    testModifyOrder();
    testDeleteOrder();

    std::cout << "All unit tests passed.\n\n";
}

double benchmarkBaseImplementation(const std::vector<OptimizedOrder>& orders) {
    OrderBook book;

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& order : orders) {
        book.addOrder(order.id, order.price, order.quantity, order.isBuy);
    }

    auto end = std::chrono::high_resolution_clock::now();

    assert(book.getOrderCount() == static_cast<int>(orders.size()));

    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

double benchmarkOptimizedImplementation(const std::vector<OptimizedOrder>& orders) {
    OptimizedOrderBook book(orders.size());

    auto start = std::chrono::high_resolution_clock::now();

    book.processOrdersBatch(orders);

    auto end = std::chrono::high_resolution_clock::now();

    assert(book.getOrderCount() == static_cast<int>(orders.size()));

    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

void runScalabilityBenchmark() {
    std::vector<int> orderSizes = {
        1000, 5000, 10000, 50000, 100000, 500000, 1000000
    };

    std::ofstream outFile("benchmark_results.csv");
    outFile << "orders,base_seconds,optimized_seconds,speedup\n";

    std::cout << "HFT Order Book Scalability Benchmark\n";
    std::cout << "====================================\n";
    std::cout << std::fixed << std::setprecision(6);

    for (int numOrders : orderSizes) {
        std::cout << "Testing " << numOrders << " orders...\n";

        std::vector<OptimizedOrder> orders = generateOrders(numOrders);

        double baseTime = benchmarkBaseImplementation(orders);
        double optimizedTime = benchmarkOptimizedImplementation(orders);
        double speedup = baseTime / optimizedTime;

        std::cout << "Base time:      " << baseTime << " seconds\n";
        std::cout << "Optimized time: " << optimizedTime << " seconds\n";
        std::cout << "Speedup:        " << speedup << "x\n\n";

        outFile << numOrders << ","
                << baseTime << ","
                << optimizedTime << ","
                << speedup << "\n";
    }

    outFile.close();

}

int main() {
    profileBaseImplementation();
    runUnitTests();
    runScalabilityBenchmark();

    return 0;
}