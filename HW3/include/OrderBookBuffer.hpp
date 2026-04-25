#pragma once
#include <iostream>
#include <cstddef>
#include <mutex>
#include <cstring>
#include <stdexcept>

// Allocator Policies

struct HeapAllocator {
    template <typename U>
    U* allocate(std::size_t n) {
        return new U[n];
    }

    template <typename U>
    void deallocate(U* ptr) {
        delete[] ptr;
    }
};

struct StackAllocator {
    // simplified stack buffer 
    char buffer[8192];
    std::size_t offset = 0;

    template <typename U>
    U* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(U);
        if (offset + bytes > sizeof(buffer)) {
            throw std::out_of_range("StackAllocator out of memory");
        }
        U* ptr = reinterpret_cast<U*>(buffer + offset);
        offset += bytes;
        return ptr;
    }

    template <typename U>
    void deallocate(U*) {
    }
};


struct ZeroInitAllocator {
    template <typename U>
    U* allocate(std::size_t n) {
        U* ptr = new U[n];
        std::memset(ptr, 0, n * sizeof(U));
        return ptr;
    }

    template <typename U>
    void deallocate(U* ptr) {
        delete[] ptr;
    }
};


//  Threading Policies

struct MutexLock {
    std::mutex mtx;
    void lock() { mtx.lock(); }
    void unlock() { mtx.unlock(); }
};

struct NoLock {
    void lock() {}
    void unlock() {}
};


// Policy Based Container

template <typename T, typename AllocatorPolicy, typename ThreadingPolicy>
class OrderBookBuffer {
private:
    AllocatorPolicy allocator_;
    ThreadingPolicy lock_;
    T* data_;
    std::size_t capacity_;
    std::size_t size_;

public:
    explicit OrderBookBuffer(std::size_t cap) : capacity_(cap), size_(0) {
        data_ = allocator_.template allocate<T>(capacity_);
    }

    ~OrderBookBuffer() {
        allocator_.template deallocate<T>(data_);
    }

    void add_order(const T& item) {
        lock_.lock();
        if (size_ < capacity_) {
            data_[size_] = item;
            ++size_;
        }
        lock_.unlock();
    }

    void print_orders() {
        lock_.lock();
        for (std::size_t i = 0; i < size_; ++i) {
            std::cout << "  -> Order ID: " << data_[i].id 
                      << " | Price: " << data_[i].price 
                      << " | Qty: " << data_[i].qty << "\n";
        }
        lock_.unlock();
    }
};