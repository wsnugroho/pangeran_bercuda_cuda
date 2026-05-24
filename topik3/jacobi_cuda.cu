/*
 * jacobi_cuda.cu
 * Topik 3A: Jacobi Iteration (CUDA)
 *
 * Compile:
 *   nvcc -O2 -allow-unsupported-compiler -o jacobi_cuda jacobi_cuda.cu
 *
 * Run:
 *   ./jacobi_cuda [N] [BLOCK_SIZE]
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_N 512
#define DEFAULT_BLOCK_SIZE 256
#define TOL 1e-6
#define MAX_ITER 5000
#define SEED 42ULL
#define DIAG_SCALE 4.0
#define REDUCE_THREADS 256

#define CUDA_CHECK(call) do {                                                \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(err__));              \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static unsigned long long next_random(unsigned long long *state) {
    *state = (*state * 2862933555777941757ULL) + 3037000493ULL;
    return *state;
}

static double random_unit(unsigned long long *state) {
    return (double)(next_random(state) >> 11) * (1.0 / 9007199254740992.0);
}

static double *alloc_vector(int n) {
    double *ptr = (double *)malloc((size_t)n * sizeof(double));
    if (ptr == NULL) {
        fprintf(stderr, "malloc gagal untuk vector %d elemen.\n", n);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static double *alloc_matrix(int n) {
    double *ptr = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (ptr == NULL) {
        fprintf(stderr, "malloc gagal untuk matrix %d x %d.\n", n, n);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static void generate_system(double *A, double *b, double *x_ref, int N) {
    unsigned long long state = SEED;

    for (int i = 0; i < N; i++) {
        x_ref[i] = 1.0;
    }

    for (int i = 0; i < N; i++) {
        double row_sum = 0.0;
        size_t row_offset = (size_t)i * (size_t)N;

        for (int j = 0; j < N; j++) {
            if (i == j) {
                A[row_offset + (size_t)j] = 0.0;
                continue;
            }

            A[row_offset + (size_t)j] = random_unit(&state);
            row_sum += fabs(A[row_offset + (size_t)j]);
        }

        A[row_offset + (size_t)i] = DIAG_SCALE * row_sum + 1.0;
        b[i] = row_sum + A[row_offset + (size_t)i];
    }
}

static double compute_inf_error(const double *x, const double *x_ref, int N) {
    double max_err = 0.0;

    for (int i = 0; i < N; i++) {
        double err = fabs(x[i] - x_ref[i]);
        if (err > max_err) {
            max_err = err;
        }
    }

    return max_err;
}

static double compute_inf_residual(const double *A, const double *x,
                                   const double *b, int N) {
    double max_res = 0.0;

    for (int i = 0; i < N; i++) {
        double ax = 0.0;
        size_t row_offset = (size_t)i * (size_t)N;

        for (int j = 0; j < N; j++) {
            ax += A[row_offset + (size_t)j] * x[j];
        }

        {
            double res = fabs(ax - b[i]);
            if (res > max_res) {
                max_res = res;
            }
        }
    }

    return max_res;
}

__global__ static void jacobi_step_kernel(const double *A, const double *b,
                                          const double *x_curr, double *x_next,
                                          double *diffs, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) {
        return;
    }

    double sigma = 0.0;
    size_t row_offset = (size_t)row * (size_t)N;
    for (int j = 0; j < N; j++) {
        if (j != row) {
            sigma += A[row_offset + (size_t)j] * x_curr[j];
        }
    }

    x_next[row] = (b[row] - sigma) / A[row_offset + (size_t)row];
    diffs[row] = fabs(x_next[row] - x_curr[row]);
}

__global__ static void reduce_max_kernel(const double *input, double *output, int n) {
    __shared__ double sdata[REDUCE_THREADS];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2U) + threadIdx.x;
    double value = 0.0;

    if (i < (unsigned int)n) {
        value = input[i];
    }
    if (i + blockDim.x < (unsigned int)n) {
        double other = input[i + blockDim.x];
        if (other > value) {
            value = other;
        }
    }

    sdata[tid] = value;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (tid < stride && sdata[tid + stride] > sdata[tid]) {
            sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0U) {
        output[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    int block_size = (argc > 2) ? atoi(argv[2]) : DEFAULT_BLOCK_SIZE;
    if (N <= 0) {
        fprintf(stderr, "N harus > 0.\n");
        return EXIT_FAILURE;
    }
    if (block_size <= 0) {
        fprintf(stderr, "BLOCK_SIZE harus > 0.\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (block_size > prop.maxThreadsPerBlock) {
        fprintf(stderr, "BLOCK_SIZE=%d melebihi maxThreadsPerBlock=%d.\n",
                block_size, prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    int grid_size = (N + block_size - 1) / block_size;

    double *h_A = alloc_matrix(N);
    double *h_b = alloc_vector(N);
    double *h_x_ref = alloc_vector(N);
    double *h_x = alloc_vector(N);
    generate_system(h_A, h_b, h_x_ref, N);
    memset(h_x, 0, (size_t)N * sizeof(double));

    double *d_A = NULL;
    double *d_b = NULL;
    double *d_x_curr = NULL;
    double *d_x_next = NULL;
    double *d_diff_a = NULL;
    double *d_diff_b = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_A, (size_t)N * (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_x_curr, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_x_next, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_diff_a, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_diff_b, (size_t)N * sizeof(double)));

    double t_comm = 0.0;
    double t_compute = 0.0;

    printf("============================================================\n");
    printf("  Topik 3A - Jacobi Iteration (CUDA)\n");
    printf("  GPU=%s  N=%d  BLOCK_SIZE=%d  GRID_SIZE=%d\n",
           prop.name, N, block_size, grid_size);
    printf("  TOL=%.1e  MAX_ITER=%d  SEED=%llu\n",
           TOL, MAX_ITER, (unsigned long long)SEED);
    printf("============================================================\n");

    CUDA_CHECK(cudaMemset(d_x_curr, 0, (size_t)N * sizeof(double)));
    {
        double t0 = now_seconds();
        CUDA_CHECK(cudaMemcpy(d_A, h_A, (size_t)N * (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
        t_comm += now_seconds() - t0;
    }

    int iterations = 0;
    int converged = 0;
    double last_diff = 0.0;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        double *reduce_in = d_diff_a;
        double *reduce_out = d_diff_b;
        int reduce_n = N;

        {
            double t0 = now_seconds();
            jacobi_step_kernel<<<grid_size, block_size>>>(d_A, d_b, d_x_curr, d_x_next, d_diff_a, N);

            while (reduce_n > 1) {
                int blocks = (reduce_n + (REDUCE_THREADS * 2) - 1) / (REDUCE_THREADS * 2);
                reduce_max_kernel<<<blocks, REDUCE_THREADS>>>(reduce_in, reduce_out, reduce_n);
                reduce_n = blocks;

                {
                    double *tmp = reduce_in;
                    reduce_in = reduce_out;
                    reduce_out = tmp;
                }
            }

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            t_compute += now_seconds() - t0;
        }

        {
            double t0 = now_seconds();
            CUDA_CHECK(cudaMemcpy(&last_diff, reduce_in, sizeof(double), cudaMemcpyDeviceToHost));
            t_comm += now_seconds() - t0;
        }

        {
            double *tmp = d_x_curr;
            d_x_curr = d_x_next;
            d_x_next = tmp;
        }

        iterations = iter + 1;
        if (last_diff < TOL) {
            converged = 1;
            break;
        }
    }

    {
        double t0 = now_seconds();
        CUDA_CHECK(cudaMemcpy(h_x, d_x_curr, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
        t_comm += now_seconds() - t0;
    }

    {
        double error = compute_inf_error(h_x, h_x_ref, N);
        double residual = compute_inf_residual(h_A, h_x, h_b, N);

        printf("  Grid                : %d blocks\n", grid_size);
        printf("  Block               : %d threads\n", block_size);
        printf("  Status              : %s\n", converged ? "CONVERGED" : "MAX_ITER");
        printf("  Iterations          : %d\n", iterations);
        printf("  Final max diff      : %.6e\n", last_diff);
        printf("  Execution time (x)  : %.6f s\n", t_compute);
        printf("  Communication time (y): %.6f s\n", t_comm);
        printf("  ||x - x_ref||_inf   : %.6e\n", error);
        printf("  ||Ax - b||_inf      : %.6e\n", residual);
        printf("  x[0]                : %.6f\n", h_x[0]);
        printf("  x[N-1]              : %.6f\n", h_x[N - 1]);
        printf("  FORMAT TABEL (x/y)  : (%.6f / %.6f)\n", t_compute, t_comm);
        printf("SUMMARY mode=cuda label=grid=%d_block=%d N=%d x=%.6f y=%.6f iter=%d residual=%.6e error=%.6e\n",
               grid_size, block_size, N, t_compute, t_comm, iterations, residual, error);
    }

    printf("============================================================\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x_curr));
    CUDA_CHECK(cudaFree(d_x_next));
    CUDA_CHECK(cudaFree(d_diff_a));
    CUDA_CHECK(cudaFree(d_diff_b));

    free(h_A);
    free(h_b);
    free(h_x_ref);
    free(h_x);
    return EXIT_SUCCESS;
}
