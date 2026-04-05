#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <numeric>
#include <string>
#include <cstdlib>
#include <random>

// Matrix-Vector Multiplication (Row-Major)
void multiply_mv_row_major(const double* matrix, int rows, int cols, const double* vector, double* result) {
    if (matrix == nullptr || vector == nullptr || result == nullptr) {
        std::cerr << "Error: Null pointer passed to multiply_mv_row_major" << std::endl;
        return;
    }
    if (rows <= 0 || cols <= 0) {
        std::cerr << "Error: Invalid dimensions" << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        result[i] = 0; // 
        for (int j = 0; j < cols; ++j) {
            // Row-major index: i * cols + j
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

// Matrix-Vector Multiplication (Column-Major)
void multiply_mv_col_major(const double* matrix, int rows, int cols, const double* vector, double* result) {
    if (matrix == nullptr || vector == nullptr || result == nullptr) {
        std::cerr << "Error: Null pointer passed to multiply_mv_col_major" << std::endl;
        return;
    }
    if (rows <= 0 || cols <= 0) {
        std::cerr << "Error: Invalid dimensions" << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < cols; ++j) {
            // Column-major index: j * rows + i
            result[i] += matrix[j * rows + i] * vector[j];
        }
    }
}

//  Matrix-Matrix Multiplication (Naive)
void multiply_mm_naive(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result) {
    if (matrixA == nullptr || matrixB == nullptr || result == nullptr) {
        std::cerr << "Error: Null pointer passed to multiply_mm_naive" << std::endl;
        return;
    }
    if (colsA != rowsB) {
        std::cerr << "Error: Inner dimensions do not match for matrix multiplication" << std::endl;
        return;
    }

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            result[i * colsB + j] = 0.0;
            for (int k = 0; k < colsA; ++k) {
                // A[i,k], B[k,j]
                result[i * colsB + j] += matrixA[i * colsA + k] * matrixB[k * colsB + j];
            }
        }
    }
}

// Matrix-Matrix Multiplication (Transposed B)
void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA, const double* matrixB_transposed, int rowsB, int colsB, double* result) {
    if (matrixA == nullptr || matrixB_transposed == nullptr || result == nullptr) {
        std::cerr << "Error: Null pointer passed to multiply_mm_transposed_b" << std::endl;
        return;
    }
    if (colsA != rowsB) {
        std::cerr << "Error: Inner dimensions do not match" << std::endl;
        return;
    }

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            result[i * colsB + j] = 0.0;
            for (int k = 0; k < colsA; ++k) {
                result[i * colsB + j] += matrixA[i * colsA + k] * matrixB_transposed[j * colsA + k];
            }
        }
    }
}

// Helper function to print matrices for testing
void print_matrix(const double* matrix, int rows, int cols, const char* name) {
    std::cout << "--- " << name << " ---" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

//Helper function for calculating std
double calculate_std_dev(const std::vector<double>& times, double mean) {
    double variance = 0;
    for (double t : times) {
        variance += (t - mean) * (t - mean);
    }
    variance /= times.size();
    return std::sqrt(variance);
}

//Helper aligned to a specific boundary
double* allocate_aligned(size_t num_elements) {
    void* ptr = nullptr;
    // 64bytes
    if (posix_memalign(&ptr, 64, num_elements * sizeof(double)) != 0) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return nullptr; 
    }
    return static_cast<double*>(ptr);
}

//Helper deallocate
void free_aligned(double* ptr) {
    free(ptr); 
}

//Optimized Naive MM
void multiply_mm_reordered(const double* A, int rowsA, int colsA, const double* B, int rowsB, int colsB, double* C) {
    if (A == nullptr || B == nullptr || C == nullptr) {
        throw std::invalid_argument("Error: Null pointer passed to matrix multiplication.");
    }
    if (colsA != rowsB) {
        throw std::invalid_argument("Error: Matrix dimensions mismatch. colsA must equal rowsB.");
    }
    for (int i = 0; i < rowsA; ++i) {
        for (int k = 0; k < colsA; ++k) {
            double a_val = A[i * colsA + k]; 
            for (int j = 0; j < colsB; ++j) {
                C[i * colsB + j] += a_val * B[k * colsB + j];
            }
        }
    }
}

int main() {
  
    // test 1: MV Row-Major 
    int r1 = 2, c1 = 3;
    double* A_row = new double[6]{1, 2, 3, 4, 5, 6}; // 2x3 matrix
    double* V = new double[3]{1, 2, 3};// 3x1 vector
    double* res_V = new double[2]; // 2x1 result
    
    multiply_mv_row_major(A_row, r1, c1, V, res_V);
    print_matrix(res_V, r1, 1, "MV Row-Major Expected: 14, 32");

    // test 2: MV Col-Major 
    // [1 4]
    // [2 5]
    // [3 6] 
    double* A_col = new double[6]{1, 4, 2, 5, 3, 6}; 
    double* res_V2 = new double[2];
    
    multiply_mv_col_major(A_col, r1, c1, V, res_V2);
    print_matrix(res_V2, r1, 1, "MV Col-Major Expected: 14, 32");

    // test 3: MM Naive
    int rA = 2, cA = 2; // Matrix A is 2x2
    int rB = 2, cB = 3; // Matrix B is 2x3
    double* MatA = new double[4]{1, 2, 3, 4};
    double* MatB = new double[6]{1, 2, 3, 4, 5, 6};
    double* res_M = new double[rA * cB]; // 2x3 result
    
    multiply_mm_naive(MatA, rA, cA, MatB, rB, cB, res_M);
    print_matrix(res_M, rA, cB, "MM Naive Expected: [9 12 15], [19 26 33]");

    // test4: MM Transposed B 
    // [1 4]
    // [2 5]
    // [3 6]
    double* MatB_T = new double[6]{1, 4, 2, 5, 3, 6};
    double* res_M2 = new double[rA * cB]; 
    
    multiply_mm_transposed_b(MatA, rA, cA, MatB_T, rB, cB, res_M2);
    print_matrix(res_M2, rA, cB, "MM Transposed B Expected: [9 12 15], [19 26 33]");

    //deallocate
    delete[] A_row;
    delete[] V;
    delete[] res_V;
    delete[] A_col;
    delete[] res_V2;
    delete[] MatA;
    delete[] MatB;
    delete[] res_M;
    delete[] MatB_T;
    delete[] res_M2;

    //Benchmarking
    std::vector<int> matrix_sizes = {128, 256, 512, 1024}; 
    const int num_runs = 5; 

    for (int N : matrix_sizes) {
        std::cout << "========================================\n";
        std::cout << "Testing Size: N = " << N << " (" << N << "x" << N << ")\n";
        
        // innitial allocate
        // double* MatA   = new double[N * N];
        // double* MatB   = new double[N * N];
        // double* MatB_T = new double[N * N]; 
        // double* VecX   = new double[N];    
        
        // double* res_Vec = new double[N];    
        // double* res_Mat = new double[N * N]; 

        //aligned allocaiton
        double* MatA   = allocate_aligned(N * N);
        double* MatB   = allocate_aligned(N * N);
        double* MatB_T = allocate_aligned(N * N); 
        double* VecX   = allocate_aligned(N);    
        
        double* res_Vec = allocate_aligned(N);    
        double* res_Mat = allocate_aligned(N * N);

        for(int i = 0; i < N * N; ++i) {
            MatA[i] = 1; 
            MatB[i] = 2; 
            MatB_T[i] = 2; // 
        }
        for(int i = 0; i < N; ++i) {
            VecX[i] = 3;
        }
        
        //helper for run test
        auto run_benchmark = [&](const std::string& name, auto test_func) {
            std::vector<double> times;
            for (int run = 0; run < num_runs; ++run) {
                if (run == 0) test_func(); // warm up 

                auto start_time = std::chrono::high_resolution_clock::now();
                test_func();
                auto end_time = std::chrono::high_resolution_clock::now();
                
                std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
                times.push_back(elapsed.count());
            }
            double sum = std::accumulate(times.begin(), times.end(), 0.0);
            double mean_time = sum / times.size();
            double std_dev = calculate_std_dev(times, mean_time);
            std::cout << "[" << name << "] Avg: " << mean_time << " ms | StdDev: " << std_dev << " ms\n";
        };

        // apply test 
        run_benchmark("MV Row-Major", [&](){ multiply_mv_row_major(MatA, N, N, VecX, res_Vec); });
        run_benchmark("MV Col-Major", [&](){ multiply_mv_col_major(MatA, N, N, VecX, res_Vec); });
        run_benchmark("MM Naive      ", [&](){ multiply_mm_naive(MatA, N, N, MatB, N, N, res_Mat); });
        run_benchmark("MM Transposed ", [&](){ multiply_mm_transposed_b(MatA, N, N, MatB_T, N, N, res_Mat); });

        // //initial  deallocate
        // delete[] MatA;
        // delete[] MatB;
        // delete[] MatB_T;
        // delete[] VecX;
        // delete[] res_Vec;
        // delete[] res_Mat;

        //aligned deallocate
        free_aligned(MatA);
        free_aligned(MatB);
        free_aligned(MatB_T);
        free_aligned(VecX);
        free_aligned(res_Vec);
        free_aligned(res_Mat);
        
    }
    
    //Optimized Comparision
    const int N = 1024;
    std::cout << "Testing Matrix Multiplication for " << N << " x " << N << "...\n";

    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C_naive(N * N, 0.0);
    std::vector<double> C_reordered(N * N, 0.0);

    std::mt19937 gen(42); // fix seed
    std::uniform_real_distribution<double> dist(0, 1);
    for (int i = 0; i < N * N; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    // naive
    auto start_naive = std::chrono::high_resolution_clock::now();
    multiply_mm_naive(A.data(), N, N, B.data(), N, N, C_naive.data());
    auto end_naive = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_naive = end_naive - start_naive;
    std::cout << "Naive (i-j-k) Time:     " << diff_naive.count() << " seconds\n";

    //optimized
    auto start_reordered = std::chrono::high_resolution_clock::now();
    multiply_mm_reordered(A.data(), N, N, B.data(), N, N, C_reordered.data());
    auto end_reordered = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_reordered = end_reordered - start_reordered;
    std::cout << "Reordered (i-k-j) Time: " << diff_reordered.count() << " seconds\n";

    std::cout << "Speedup:                " << diff_naive.count() / diff_reordered.count() << "x faster!\n";



    return 0;
}