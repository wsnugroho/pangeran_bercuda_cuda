/*
 * jacobi_mpi.c
 * Jacobi Iteration — MPI (Multicore / Cluster)
 * Topik 3: Pembanding platform paralel untuk Jacobi
 *
 * Kompilasi: mpicc -O2 -o jacobi_mpi jacobi_mpi.c -lm
 * Jalankan : mpirun -np 4  ./jacobi_mpi [N]
 *            mpirun -np 8  ./jacobi_mpi [N]
 *            mpirun -np 16 ./jacobi_mpi [N]
 *
 * Strategi:
 *   - Rank 0 membagi baris matriks A ke semua proses (Scatterv).
 *   - b dibroadcast penuh ke semua rank.
 *   - x diinisialisasi 0 di semua rank.
 *   - Tiap iterasi:
 *       1. Setiap rank hitung x_new untuk baris lokal.
 *       2. MPI_Allgatherv x_new agar semua rank punya vektor penuh.
 *       3. Setiap rank hitung local_max_diff.
 *       4. MPI_Allreduce(MPI_MAX) untuk global convergence check.
 *   - Pengukuran: komputasi vs komunikasi terpisah.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

float *allocMatrix(int rows, int cols) {
    float *m = (float *)calloc(rows * cols, sizeof(float));
    if (!m) { fprintf(stderr, "calloc gagal!\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    return m;
}

float *allocVector(int N) {
    float *v = (float *)calloc(N, sizeof(float));
    if (!v) { fprintf(stderr, "calloc gagal!\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    return v;
}

void initSystem(float *A, float *b, int N) {
    srand(42);
    for (int i = 0; i < N; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                A[i*N + j] = (float)rand() / RAND_MAX;
                row_sum += fabsf(A[i*N + j]);
            }
        }
        A[i*N + i] = 1.0f + row_sum;
        b[i] = (float)rand() / RAND_MAX;
    }
}

int rowsForRank(int rank, int np, int N) {
    int base  = N / np;
    int extra = N % np;
    return base + (rank < extra ? 1 : 0);
}

int offsetForRank(int rank, int np, int N) {
    int off = 0;
    for (int r = 0; r < rank; r++) off += rowsForRank(r, np, N);
    return off;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int N = (argc > 1) ? atoi(argv[1]) : 512;
    float eps = 1e-6f;
    int max_iter = 10000;

    if (rank == 0) {
        printf("============================================================\n");
        printf("  Jacobi Iteration — MPI  |  NP=%d  N=%d\n", np, N);
        printf("  Epsilon        : %e\n", eps);
        printf("  Max Iterasi    : %d\n", max_iter);
        printf("============================================================\n");
    }

    /* ─── Alokasi global di rank 0 ─── */
    float *A = NULL, *b = NULL, *x = NULL;
    if (rank == 0) {
        A = allocMatrix(N, N);
        b = allocVector(N);
        x = allocVector(N);
        initSystem(A, b, N);
    }

    /* ─── Scatter counts dan displacements untuk A (per baris) ─── */
    int *sendCounts = (int *)malloc(np * sizeof(int));
    int *displs     = (int *)malloc(np * sizeof(int));
    int *recvCounts = (int *)malloc(np * sizeof(int));
    int *recvDispls = (int *)malloc(np * sizeof(int));
    for (int r = 0; r < np; r++) {
        sendCounts[r] = rowsForRank(r, np, N) * N;
        displs[r]     = offsetForRank(r, np, N) * N;
        recvCounts[r] = rowsForRank(r, np, N);
        recvDispls[r] = offsetForRank(r, np, N);
    }

    int myRows  = rowsForRank(rank, np, N);
    int myStart = offsetForRank(rank, np, N);

    float *localA = allocMatrix(myRows, N);
    float *localB = allocVector(N);
    float *x_old  = allocVector(N);
    float *x_new  = allocVector(N);

    /* ─── Komunikasi awal: Scatter A + Broadcast b ─── */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_comm_start = MPI_Wtime();

    MPI_Scatterv(A, sendCounts, displs, MPI_FLOAT,
                 localA, myRows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank == 0) memcpy(localB, b, N * sizeof(float));
    MPI_Bcast(localB, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double t_comm_init = MPI_Wtime() - t_comm_start;

    /* ─── Iterasi Jacobi ─── */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_compute_start = MPI_Wtime();

    int iter = 0;
    float global_max_diff = eps + 1.0f;
    double t_comm_iter = 0.0;

    while (iter < max_iter && global_max_diff >= eps) {
        /* Komputasi lokal */
        float local_max_diff = 0.0f;
        for (int i = 0; i < myRows; i++) {
            int gi = myStart + i;
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                if (j != gi) sum += localA[i*N + j] * x_old[j];
            }
            x_new[gi] = (localB[gi] - sum) / localA[i*N + gi];
            float diff = fabsf(x_new[gi] - x_old[gi]);
            if (diff > local_max_diff) local_max_diff = diff;
        }

        /* Komunikasi per-iterasi */
        double t_iter_comm_start = MPI_Wtime();

        /* Allgatherv: semua rank punya x_new penuh */
        MPI_Allgatherv(x_new + myStart, myRows, MPI_FLOAT,
                       x_new, recvCounts, recvDispls, MPI_FLOAT,
                       MPI_COMM_WORLD);

        /* Allreduce max diff */
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_FLOAT, MPI_MAX,
                      MPI_COMM_WORLD);

        t_comm_iter += MPI_Wtime() - t_iter_comm_start;

        /* Swap x_old dan x_new (hanya pointer) */
        float *tmp = x_old; x_old = x_new; x_new = tmp;
        iter++;
    }

    double t_compute = MPI_Wtime() - t_compute_start - t_comm_iter;
    double t_comm_total = t_comm_init + t_comm_iter;

    /* ─── Reduce timing ke rank 0 ─── */
    double maxCompute, maxComm, maxCommInit, maxCommIter;
    MPI_Reduce(&t_compute,    &maxCompute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_total, &maxComm,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_init,  &maxCommInit, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_iter,  &maxCommIter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double gflops = (2.0 * N * N * iter) / (maxCompute * 1e9);

        printf("\n  ┌─────────────────────────────────────────┐\n");
        printf("  │ HASIL PENGUKURAN WAKTU (MPI, NP=%2d)     │\n", np);
        printf("  ├─────────────────────────────────────────┤\n");
        printf("  │ Komputasi (max)   : %10.6f detik    │\n", maxCompute);
        printf("  │ Komunikasi (max)  : %10.6f detik    │\n", maxComm);
        printf("  │   - Init (Scat+BC): %10.6f detik    │\n", maxCommInit);
        printf("  │   - Per-iterasi   : %10.6f detik    │\n", maxCommIter);
        printf("  │ Iterasi           : %10d          │\n", iter);
        printf("  │ GFLOPS            : %10.4f          │\n", gflops);
        printf("  ├─────────────────────────────────────────┤\n");
        printf("  │ FORMAT TABEL (x/y):                     │\n");
        printf("  │   x = %.6f s  y = %.6f s       │\n", maxCompute, maxComm);
        printf("  └─────────────────────────────────────────┘\n");

        /* Verifikasi vs referensi CPU */
        char refFile[64];
        sprintf(refFile, "ref_x_N%d.bin", N);
        FILE *fp = fopen(refFile, "rb");
        if (fp) {
            float *ref = allocVector(N);
            fread(ref, sizeof(float), N, fp); fclose(fp);
            int ok = 1;
            for (int i = 0; i < N; i++) {
                float diff = fabsf(x_old[i] - ref[i]);
                float tol  = 1e-2f + 1e-4f * fabsf(ref[i]);
                if (diff > tol) { ok = 0; break; }
            }
            printf("\n  [VERIFIKASI] %s\n", ok ? "PASS ✓" : "FAIL ✗");
            free(ref);
        } else {
            printf("\n  [VERIFIKASI] File referensi tidak ditemukan.\n");
        }

        printf("============================================================\n");
        free(A); free(b); free(x);
    }

    free(sendCounts); free(displs); free(recvCounts); free(recvDispls);
    free(localA); free(localB); free(x_old); free(x_new);
    MPI_Finalize();
    return 0;
}
