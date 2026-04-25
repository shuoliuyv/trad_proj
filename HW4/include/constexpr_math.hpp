#pragma once

// Constexpr Factorial
constexpr int factorial_calc(int n) {
    return (n <= 1) ? 1 : (n * factorial_calc(n - 1));
}

// Constexpr Fibonacci
constexpr int fibonacci_calc(int n) {
    return (n <= 0) ? 0 : (n == 1 ? 1 : fibonacci_calc(n - 1) + fibonacci_calc(n - 2));
}

// Constexpr Square
constexpr int square(int x) {
    return x * x;
}

// Constexpr Price Bucket (Rounding down to nearest 0.05)
constexpr double price_bucket(double price) {
    // resecaling trick
    return static_cast<long long>(price * 20) / 20;
}