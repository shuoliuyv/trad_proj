#pragma once

// return an iterator to the first element satisfying the predicate, or 'last' if not found.
template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first, InputIterator last, Predicate pred) {
    for (; first != last; ++first) {
        if (pred(*first)) {
            return first;
        }
    }
    return last;
}