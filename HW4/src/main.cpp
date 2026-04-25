#include <iostream>
#include "metaprogramming.hpp"
#include "constexpr_math.hpp"
#include "StaticVector.hpp"
#include "generic_algorithms.hpp"
#include "OrderBookBuffer.hpp"


// Compile-time static assertions for TMP logic
static_assert(Factorial<5>::value == 120, "Factorial<5> compilation failed.");
static_assert(Fibonacci<7>::value == 13, "Fibonacci<7> compilation failed.");

// Part 2 Constexpr 
static_assert(factorial_calc(5) == 120, "Constexpr Factorial failed.");
static_assert(fibonacci_calc(7) == 13, "Constexpr Fibonacci failed.");
static_assert(square(5) == 25, "Constexpr Square failed.");

// Validating price bucket at compile time
static_assert(price_bucket(101.73) == 101.70, "Constexpr Price Bucket failed.");


// Part 3 Order Structure defin
struct Order {
    int id;
    double price;
    int qty;
};

int main() {
    std::cout << "🚀 HFT Template Homework Starter\n";

    std::cout << "Part 1 Test\n";

    // Test print_if_odd
    std::cout << "[SFINAE Test]\n";
    print_if_odd(7);  // will print
    print_if_odd(4);  // will not 
    
    // print_if_odd(3.14); //  will cause a compile error

    // Test print_all
    std::cout << "\n[Variadic Template Test]\n";
    print_all("Order_ID:", 1001, "Price:", 150.25, "Status:", 'A');


    std::cout << "\nPart 2 Test\n";
    // Using a constexpr value to size an array 
    constexpr int Size = square(5);
    int arr[Size]; 
    
    // initialize array to prevent unused variable warning
    for(int i = 0; i < Size; ++i) {
        arr[i] = i;
    }
    
    std::cout << "Array successfully created on stack with constexpr size: " << Size << "\n";
    std::cout << "Runtime check of price_bucket(101.73): " << price_bucket(101.73) << "\n";

    std::cout << "\nPart 3 Test\n";
    
    // instantiate StaticVector 
    StaticVector<Order, 10> orders;
    orders.push_back({1, 95.50, 15});
    orders.push_back({2, 105.20, 20});
    orders.push_back({3, 99.00, 7});
    orders.push_back({4, 110.00, 30});

    std::cout << "Total orders added: " << orders.size() << "\n";

    // Find all orders with price > 100
    std::cout << "Orders with price > 100:\n";
    auto it = orders.begin();
    while (it != orders.end()) {
        it = find_if(it, orders.end(), [](const Order& o) { return o.price > 100.0; });
        if (it != orders.end()) {
            std::cout << "  -> Order ID: " << it->id << " | Price: " << it->price << "\n";
            ++it; // move past the found element to find the next one
        }
    }

    // Find the first order with quantity divisible by 10
    std::cout << "First order with quantity divisible by 10:\n";
    auto div10_it = find_if(orders.begin(), orders.end(), [](const Order& o) { return o.qty % 10 == 0; });
    if (div10_it != orders.end()) {
        std::cout << "  -> Order ID: " << div10_it->id << " | Qty: " << div10_it->qty << "\n";
    } else {
        std::cout << "  -> Not found.\n";
    }

    std::cout << "\nPart 4 Test\n";
    
    // Configuration 1
    std::cout << "[Config 1: StackAllocator + NoLock]\n";
    OrderBookBuffer<Order, StackAllocator, NoLock> book1(10);
    book1.add_order({100, 150.00, 50});
    book1.add_order({101, 150.05, 100});
    book1.print_orders();

    // Configuration 2
    std::cout << "\n[Config 2: HeapAllocator + MutexLock]\n";
    OrderBookBuffer<Order, HeapAllocator, MutexLock> book2(10);
    book2.add_order({200, 200.50, 10});
    book2.add_order({201, 200.45, 20});
    book2.print_orders();

    // Configuration 3
    std::cout << "\n[Config 3: ZeroInitAllocator + NoLock]\n";
    OrderBookBuffer<Order, ZeroInitAllocator, NoLock> book3(2);
    book3.add_order({300, 305.10, 500});
    book3.print_orders();


    return 0;
}
