#include "../include/TradeLogger.hpp"
#include <iostream>

TradeLogger::TradeLogger(const std::string& filename, size_t batch_size) 
    : log_filename(filename), max_batch_size(batch_size) {
    
    log_file.open(log_filename, std::ios::out | std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Error: Failed to open trade log file: " << log_filename << "\n";
    }
    
    // Pre-allocate memory 
    trade_batch.reserve(max_batch_size);
}

TradeLogger::~TradeLogger() {
    // write any remaining trades in the buffer before destroying the object
    flush();
    if (log_file.is_open()) {
        log_file.close();
    }
}

void TradeLogger::log_trade(const Trade& trade) {
    trade_batch.push_back(trade);
    
    // If buffer reaches the batch size, write to disk
    if (trade_batch.size() >= max_batch_size) {
        flush();
    }
}

void TradeLogger::flush() {
    if (trade_batch.empty() || !log_file.is_open()) {
        return;
    }

    for (const auto& t : trade_batch) {
        log_file << "Symbol: " << t.symbol 
                 << " | BuyerID: " << t.buyer_id 
                 << " | SellerID: " << t.seller_id 
                 << " | Price: " << t.price 
                 << " | Qty: " << t.quantity 
                 << " | Latency(ns): " << t.latency_ns << "\n";
    }
    
    // Ensure data is written to the physical disk
    log_file.flush(); 
    
    // Clear the buffer
    trade_batch.clear(); 
}