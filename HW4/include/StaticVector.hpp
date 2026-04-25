#pragma once
#include <cstddef>
#include <stdexcept>

template <typename T, std::size_t N>
class StaticVector {
private:
    T data_[N];
    std::size_t size_;

public:
    StaticVector() : size_(0) {}

    // Add an element to the end. Throws if capacity is exceeded.
    void push_back(const T& value) {
        if (size_ >= N) {
            throw std::out_of_range("StaticVector capacity exceeded.");
        }
        data_[size_] = value;
        ++size_;
    }

    // element access
    T& operator[](std::size_t index) {
        return data_[index];
    }

    const T& operator[](std::size_t index) const {
        return data_[index];
    }

    // current number of elements
    std::size_t size() const {
        return size_;
    }

    // iterator support for algorithms (using raw pointers)
    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }
};