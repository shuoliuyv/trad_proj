#include "feed_parser.h"
#include "market_snapshot.h"
#include "order_manager.h"
#include <iostream>

int main() {
    MarketSnapshot snapshot;
    OrderManager order_manager;

    auto feed = load_feed("sample_feed.txt");

    for (const auto& event : feed) {
        event.print(); 

        // integrate with components:
        if (event.type == FeedType::BID) {
            snapshot.update_bid(event.price, event.quantity);
        } else if (event.type == FeedType::ASK) {
            snapshot.update_ask(event.price, event.quantity);
        } else if (event.type == FeedType::EXECUTION) {
            order_manager.handle_fill(event.order_id, event.quantity);
        }

        // strategy Loop 
        // If best bid is better than 100, place a SELL order.
        const PriceLevel* best_bid = snapshot.get_best_bid();
        static bool order_placed = false; 

        if (best_bid && best_bid->price > 100 && !order_placed) {
            std::cout << "[Strategy] Placing SELL order at " << best_bid->price << "\n";
            order_manager.place_order(Side::Sell, best_bid->price, 50);
            order_placed = true; // Make sure we only place it once
        }
    }

    // print final active orders
    order_manager.print_active_orders();

    return 0;
}