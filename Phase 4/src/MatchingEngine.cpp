#include "../include/MatchingEngine.hpp"
#include <algorithm>

template <typename PriceType, typename OrderIdType>
MatchingEngine<PriceType, OrderIdType>::MatchingEngine(OrderBook<PriceType, OrderIdType>& book) 
    : order_book(book) {}

template <typename PriceType, typename OrderIdType>
std::vector<Trade> MatchingEngine<PriceType, OrderIdType>::match_order(
    OrderPtr incoming_order, std::chrono::high_resolution_clock::time_point tick_time) {
    
    std::vector<Trade> executed_trades;

    if (incoming_order->is_buy) {
        // Buy order- iterate through asks
        auto it = order_book.asks.begin();
        while (it != order_book.asks.end() && incoming_order->quantity > 0) {
            // if buy price >= sell price
            if (incoming_order->price >= it->first) {
                int trade_qty = std::min(incoming_order->quantity, it->second->quantity);
                
                // calculate Tick-to-Trade Latency
                auto trade_time = std::chrono::high_resolution_clock::now();
                long long latency = std::chrono::duration_cast<std::chrono::nanoseconds>(trade_time - tick_time).count();

                // generate Trade record
                executed_trades.push_back({
                    incoming_order->id,    
                    it->second->id,        
                    incoming_order->symbol,
                    it->first,             
                    trade_qty,
                    latency
                });

                incoming_order->quantity -= trade_qty;
                it->second->quantity -= trade_qty;

                if (it->second->quantity == 0) {
                    it = order_book.asks.erase(it); // remove fully filled resting order
                } else {
                    ++it;
                }
            } else {
                break; 
            }
        }
    } else {
        // sell order- iterate through bids
        auto it = order_book.bids.begin();
        while (it != order_book.bids.end() && incoming_order->quantity > 0) {
            if (incoming_order->price <= it->first) {
                int trade_qty = std::min(incoming_order->quantity, it->second->quantity);
                
                auto trade_time = std::chrono::high_resolution_clock::now();
                long long latency = std::chrono::duration_cast<std::chrono::nanoseconds>(trade_time - tick_time).count();

                // trade record
                executed_trades.push_back({
                    it->second->id,        
                    incoming_order->id,    
                    incoming_order->symbol,
                    it->first,             
                    trade_qty,
                    latency
                });

                incoming_order->quantity -= trade_qty;
                it->second->quantity -= trade_qty;

                if (it->second->quantity == 0) {
                    it = order_book.bids.erase(it); 
                } else {
                    ++it;
                }
            } else {
                break; 
            }
        }
    }

    return executed_trades;
}

// Explicit instantiations
template class MatchingEngine<double, int>;
