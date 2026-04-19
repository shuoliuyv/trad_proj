#pragma once

#include <map>
#include <memory>
#include <iostream>

enum class Side { Buy, Sell };

enum class OrderStatus { New, Filled, PartiallyFilled, Cancelled };

// order details
struct MyOrder {
    int id;
    Side side;
    double price;
    int quantity;
    int filled = 0;
    OrderStatus status = OrderStatus::New;
};

class OrderManager {
private:
    // active orders
    std::map<int, std::unique_ptr<MyOrder>> orders;
    
    // incrementing ID
    int next_order_id = 1;

public:
    // create new order
    int place_order(Side side, double price, int qty);
    
    // remove order
    void cancel(int id);
    
    // update fill status
    void handle_fill(int id, int filled_qty);
    
    void print_active_orders() const;
};