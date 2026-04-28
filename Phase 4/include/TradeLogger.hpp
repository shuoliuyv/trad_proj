#pragma once

#include "MatchingEngine.hpp" 
#include <vector>
#include <string>
#include <fstream>

class TradeLogger {
public:
    // RAII: open the file upon creation
    TradeLogger(const std::string& filename, size_t batch_size = 1000);
    
    // RAII: flushed upon destruction
    ~TradeLogger(); 

    // add a trade
    void log_trade(const Trade& trade);

    // Force write the current batch to disk
    void flush();

private:
    std::string log_filename;
    std::ofstream log_file;
    size_t max_batch_size;
    
    // Buffer to hold trades before batch writing
    std::vector<Trade> trade_batch; 
};