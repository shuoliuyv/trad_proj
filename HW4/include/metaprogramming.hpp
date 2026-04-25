#pragma once
#include <type_traits>
#include <iostream>

// TMP-based factorial
template<int N> struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};
template<> struct Factorial<0> {
    static constexpr int value = 1;
};

// Add more TMP utilities, traits, and enable_if functions here
// TMP Fibonacci
template<int N> 
struct Fibonacci {
    static constexpr int value = Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template<> 
struct Fibonacci<0> {
    static constexpr int value = 0;
};

template<> 
struct Fibonacci<1> {
    static constexpr int value = 1;
};


// IsOdd Trait & SFINAE print_if_odd
template <typename T, typename Enable = void>
struct IsOdd : std::false_type {};

template <typename T>
struct IsOdd<T, typename std::enable_if<std::is_integral<T>::value>::type> : std::true_type {};

template <typename T>
typename std::enable_if<IsOdd<T>::value, void>::type
print_if_odd(T value) {
    if (value % 2 != 0) {
        std::cout << "Value " << value << " is odd.\n";
    }
}

// Variadic Template print_all
template <typename... Args>
void print_all(Args... args) {
    // Fold expression over the comma operator
    ((std::cout << args << " "), ...);
    std::cout << "\n";
}