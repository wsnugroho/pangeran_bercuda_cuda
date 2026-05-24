/*
 * jacobi_mpi.c
 * Topik 3A: Jacobi Iteration (MPI)
 *
 * Compile:
 *   mpicc -O2 -o jacobi_mpi jacobi_mpi.c -lm
 *
 * Run:
 *   OMPI_MCA_coll_hcoll_enable=0 mpirun --allow-run-as-root -np <NP> ./jacobi_mpi [N]
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_N 512
#define TOL 1e-6
#define MAX_ITER 5000
#define SEED 42ULL
#define DIAG_SCALE 4.0

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
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return ptr;
}

static double *alloc_matrix_rows(int rows, int cols) {
    double *ptr = (double *)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if (ptr == NULL) {
        fprintf(stderr, "malloc gagal untuk matrix lokal %d x %d.\n", rows, cols);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return ptr;
}

static void build_counts(int N, int np, int *row_counts, int *row_displs) {
    int base = N / np;
    int extra = N % np;
    int offset = 0;

    for (int r = 0; r < np; r++) {
        row_counts[r] = base + (r < extra ? 1 : 0);
        row_displs[r] = offset;
        offset += row_counts[r];
    }
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int np = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    if (N <= 0) {
        if (rank == 0) {
            fprintf(stderr, "N harus > 0.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int *row_counts = (int *)malloc((size_t)np * sizeof(int));
    int *row_displs = (int *)malloc((size_t)np * sizeof(int));
    int *A_counts = (int *)malloc((size_t)np * sizeof(int));
    int *A_displs = (int *)malloc((size_t)np * sizeof(int));
    if (row_counts == NULL || row_displs == NULL || A_counts == NULL || A_displs == NULL) {
        fprintf(stderr, "malloc gagal untuk metadata distribusi.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    build_counts(N, np, row_counts, row_displs);
    for (int r = 0; r < np; r++) {
        A_counts[r] = row_counts[r] * N;
        A_displs[r] = row_displs[r] * N;
    }

    int my_rows = row_counts[rank];
    int my_start = row_displs[rank];

    double *A = NULL;
    double *b = NULL;
    double *x_ref = NULL;
    if (rank == 0) {
        A = alloc_matrix_rows(N, N);
        b = alloc_vector(N);
        x_ref = alloc_vector(N);
        generate_system(A, b, x_ref, N);
    }

    double *local_A = alloc_matrix_rows(my_rows, N);
    double *local_b = alloc_vector(my_rows);
    double *local_x_new = alloc_vector(my_rows);
    double *x_curr = (double *)calloc((size_t)N, sizeof(double));
    double *x_next = alloc_vector(N);
    if (x_curr == NULL) {
        fprintf(stderr, "calloc gagal untuk solusi global.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    memset(x_next, 0, (size_t)N * sizeof(double));

    if (rank == 0) {
        printf("============================================================\n");
        printf("  Topik 3A - Jacobi Iteration (MPI)\n");
        printf("  NP=%d  N=%d  TOL=%.1e  MAX_ITER=%d  SEED=%llu\n",
               np, N, TOL, MAX_ITER, (unsigned long long)SEED);
        printf("============================================================\n");
    }

    double t_comm_local = 0.0;
    double t_comp_local = 0.0;
    double t0 = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    MPI_Scatterv(A, A_counts, A_displs, MPI_DOUBLE,
                 local_A, my_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, row_counts, row_displs, MPI_DOUBLE,
                 local_b, my_rows, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    t_comm_local += MPI_Wtime() - t0;

    int iterations = 0;
    int converged = 0;
    double last_diff = 0.0;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        double local_max_diff = 0.0;

        t0 = MPI_Wtime();
        for (int i = 0; i < my_rows; i++) {
            int global_i = my_start + i;
            size_t row_offset = (size_t)i * (size_t)N;
            double sigma = 0.0;

            for (int j = 0; j < N; j++) {
                if (j != global_i) {
                    sigma += local_A[row_offset + (size_t)j] * x_curr[j];
                }
            }

            local_x_new[i] = (local_b[i] - sigma) / local_A[row_offset + (size_t)global_i];

            {
                double diff = fabs(local_x_new[i] - x_curr[global_i]);
                if (diff > local_max_diff) {
                    local_max_diff = diff;
                }
            }
        }
        t_comp_local += MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        MPI_Allgatherv(local_x_new, my_rows, MPI_DOUBLE,
                       x_next, row_counts, row_displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        MPI_Allreduce(&local_max_diff, &last_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        t_comm_local += MPI_Wtime() - t0;

        {
            double *tmp = x_curr;
            x_curr = x_next;
            x_next = tmp;
        }

        iterations = iter + 1;
        if (last_diff < TOL) {
            converged = 1;
            break;
        }
    }

    {
        double max_comp = 0.0;
        double max_comm = 0.0;
        MPI_Reduce(&t_comp_local, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&t_comm_local, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double error = compute_inf_error(x_curr, x_ref, N);
            double residual = compute_inf_residual(A, x_curr, b, N);

            printf("  Status              : %s\n", converged ? "CONVERGED" : "MAX_ITER");
            printf("  Iterations          : %d\n", iterations);
            printf("  Final max diff      : %.6e\n", last_diff);
            printf("  Execution time (x)  : %.6f s\n", max_comp);
            printf("  Communication time (y): %.6f s\n", max_comm);
            printf("  ||x - x_ref||_inf   : %.6e\n", error);
            printf("  ||Ax - b||_inf      : %.6e\n", residual);
            printf("  x[0]                : %.6f\n", x_curr[0]);
            printf("  x[N-1]              : %.6f\n", x_curr[N - 1]);
            printf("  FORMAT TABEL (x/y)  : (%.6f / %.6f)\n", max_comp, max_comm);
            printf("SUMMARY mode=mpi label=NP=%d N=%d x=%.6f y=%.6f iter=%d residual=%.6e error=%.6e\n",
                   np, N, max_comp, max_comm, iterations, residual, error);
            printf("============================================================\n");
        }
    }

    free(row_counts);
    free(row_displs);
    free(A_counts);
    free(A_displs);
    free(local_A);
    free(local_b);
    free(local_x_new);
    free(x_curr);
    free(x_next);
    if (rank == 0) {
        free(A);
        free(b);
        free(x_ref);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
