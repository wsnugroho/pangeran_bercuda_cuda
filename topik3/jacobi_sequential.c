/*
 * jacobi_sequential.c
 * Topik 3A: Jacobi Iteration (Sequential CPU)
 *
 * Compile:
 *   gcc -O2 -o jacobi_sequential jacobi_sequential.c -lm
 *
 * Run:
 *   ./jacobi_sequential [N]
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_N 512
#define TOL 1e-6
#define MAX_ITER 5000
#define SEED 42ULL
#define DIAG_SCALE 4.0

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

        double res = fabs(ax - b[i]);
        if (res > max_res) {
            max_res = res;
        }
    }

    return max_res;
}

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    if (N <= 0) {
        fprintf(stderr, "N harus > 0.\n");
        return EXIT_FAILURE;
    }

    double *A = alloc_matrix(N);
    double *b = alloc_vector(N);
    double *x_ref = alloc_vector(N);
    double *x_curr = (double *)calloc((size_t)N, sizeof(double));
    double *x_next = alloc_vector(N);

    if (x_curr == NULL) {
        fprintf(stderr, "calloc gagal untuk solusi awal.\n");
        return EXIT_FAILURE;
    }

    generate_system(A, b, x_ref, N);
    memset(x_next, 0, (size_t)N * sizeof(double));

    printf("============================================================\n");
    printf("  Topik 3A - Jacobi Iteration (Sequential CPU)\n");
    printf("  N=%d  TOL=%.1e  MAX_ITER=%d  SEED=%llu\n",
           N, TOL, MAX_ITER, (unsigned long long)SEED);
    printf("============================================================\n");

    int iterations = 0;
    int converged = 0;
    double last_diff = 0.0;

    double t_start = now_seconds();
    for (int iter = 0; iter < MAX_ITER; iter++) {
        double max_diff = 0.0;

        for (int i = 0; i < N; i++) {
            size_t row_offset = (size_t)i * (size_t)N;
            double sigma = 0.0;

            for (int j = 0; j < N; j++) {
                if (j != i) {
                    sigma += A[row_offset + (size_t)j] * x_curr[j];
                }
            }

            x_next[i] = (b[i] - sigma) / A[row_offset + (size_t)i];

            double diff = fabs(x_next[i] - x_curr[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }

        {
            double *tmp = x_curr;
            x_curr = x_next;
            x_next = tmp;
        }

        iterations = iter + 1;
        last_diff = max_diff;
        if (max_diff < TOL) {
            converged = 1;
            break;
        }
    }
    double t_compute = now_seconds() - t_start;

    {
        double error = compute_inf_error(x_curr, x_ref, N);
        double residual = compute_inf_residual(A, x_curr, b, N);

        printf("  Status              : %s\n", converged ? "CONVERGED" : "MAX_ITER");
        printf("  Iterations          : %d\n", iterations);
        printf("  Final max diff      : %.6e\n", last_diff);
        printf("  Execution time (x)  : %.6f s\n", t_compute);
        printf("  Communication time (y): %.6f s\n", 0.0);
        printf("  ||x - x_ref||_inf   : %.6e\n", error);
        printf("  ||Ax - b||_inf      : %.6e\n", residual);
        printf("  x[0]                : %.6f\n", x_curr[0]);
        printf("  x[N-1]              : %.6f\n", x_curr[N - 1]);
        printf("  FORMAT TABEL (x/y)  : (%.6f / %.6f)\n", t_compute, 0.0);
        printf("SUMMARY mode=seq label=SEQ N=%d x=%.6f y=%.6f iter=%d residual=%.6e error=%.6e\n",
               N, t_compute, 0.0, iterations, residual, error);
    }

    printf("============================================================\n");

    free(A);
    free(b);
    free(x_ref);
    free(x_curr);
    free(x_next);
    return EXIT_SUCCESS;
}
