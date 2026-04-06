## Discussion Questions

### 1. Pointers vs. References in C++

Pointers store memory addresses and can be reassigned or set to `nullptr`. References are aliases to existing variables, so they must refer to a valid object and cannot be changed after initialization.

In numerical algorithms, pointers are more useful for low-level kernels because matrices and vectors are often stored as raw contiguous arrays, and pointer-based indexing gives direct control over memory access. References are more convenient for helper functions or higher-level code where we want cleaner syntax and know the object should always exist.

### 2. Row-Major vs. Column-Major Storage

Storage order affects how data is laid out in memory, which directly affects cache performance. In row-major storage, elements in the same row are contiguous. In column-major storage, elements in the same column are contiguous.

In our matrix-vector multiplication, the row-major version accesses
$$
A[i \cdot \text{cols} + j],
$$
so the inner loop reads contiguous memory. The column-major version accesses
$$
A[j \cdot \text{rows} + i],
$$
which leads to strided access relative to the layout used in the benchmark. That is why the row-major version becomes faster as the matrix gets larger. For example, at $$N=1024$$, the row-major version took about `3.197 ms`, while the column-major version took about `5.537 ms`.

For matrix-matrix multiplication, the naive version accesses a column of `B` in the inner loop, which is not cache-friendly in row-major storage. The transposed-`B` version changes this into contiguous access, so it performs better. At $$N=1024$$, the naive version took about `6624.3 ms`, while the transposed-`B` version took about `3847.6 ms`.

### 3. CPU Caches and Locality

CPU caches are small, fast memory layers between the processor and main memory. L1 is the fastest and smallest, L2 is larger, and L3 is larger again but slower. If data is already in cache, access is much faster than going to main memory.

Spatial locality means nearby memory locations are likely to be used soon. Temporal locality means recently used data is likely to be reused soon.

In this project, I mainly tried to exploit spatial locality by making inner loops access contiguous memory. This is why the row-major matrix-vector version and the transposed-`B` matrix-matrix version perform better. In the loop-reordered multiplication, the new loop order also improves reuse of data already brought into cache.

### 4. Memory Alignment

Memory alignment means placing data at addresses that match certain byte boundaries, such as 64-byte boundaries. This can help the CPU load data more efficiently, especially when vectorized instructions are used.

In my experiments, I used `posix_memalign` for 64-byte aligned allocation. However, I did not observe a large or consistent improvement compared with the unaligned version. The difference was small overall.

A likely reason is that alignment alone does not help much unless it is combined with stronger compiler optimization and vectorized execution.

### 5. Compiler Optimizations and Inlining

Compiler optimizations play a big role in performance. With `-O3`, the compiler can improve loops, reduce overhead, and automatically inline small functions when useful.

In my results, optimization made a clear difference. For example, at $$N=1024$$, the naive matrix-matrix multiplication improved from about `6624.3 ms` to `3659.4 ms`, and the transposed-`B` version improved from about `3847.6 ms` to `773.77 ms`.

This also shows that compiler optimization works much better when the memory access pattern is already good. Inlining can help, but in this project the main gains came from the overall optimization level.

One drawback of aggressive optimization is that debugging becomes harder, and the generated code is less straightforward to inspect.

### 6. Profiling and Bottlenecks

From profiling, the main bottleneck was matrix-matrix multiplication. The `gprof` results showed that `multiply_mm_naive` took about `82%` of the total runtime, while `multiply_mm_transposed_b` took about `17.8%`. The matrix-vector functions took very little time.

This made it clear that the main optimization effort should go into matrix-matrix multiplication. Instead of spending time on smaller parts of the code, I focused on improving cache locality in the most expensive kernel.

### 7. Teamwork Reflection

Although this assignment was designed with teamwork in mind, I completed it on my own. One thing I liked about doing it alone was that I had to work through every part myself, from the baseline implementations to benchmarking, profiling, and optimization. It took more time, but it also helped me understand the full structure of the project much better.

Doing all parts by myself made me pay closer attention to how the pieces fit together, such as storage layout, testing, benchmarking, and performance analysis. I think this hands-on process gave me a better understanding of both the code architecture and the reason behind each optimization step.

The main challenge was that everything took longer since I could not split the implementation or debugging work with others. But the benefit was that I got to see all the details directly and learned more from the full process.