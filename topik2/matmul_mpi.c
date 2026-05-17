/*
 * matmul_mpi.c
 * Perkalian Matriks Bujur Sangkar — MPI (Multicore / Multi-processor)
 * Topik 2: Pembanding platform CPU paralel
 *
 * Kompilasi: mpicc -O2 -o matmul_mpi matmul_mpi.c -lm
 * Jalankan :
 *   mpirun -np 8  ./matmul_mpi [N]   (8 prosesor)
 *   mpirun -np 16 ./matmul_mpi [N]   (16 prosesor)
 *
 * Strategi:
 *   - Rank 0 membagi baris matriks A secara merata ke semua proses.
 *   - Setiap proses menerima chunk baris A dan matriks B penuh.
 *   - Setiap proses menghitung partial rows of C secara lokal.
 *   - Rank 0 mengumpulkan semua baris C via MPI_Gather.
 *
 * Pengukuran:
 *   x = waktu komputasi (MPI_Wtime sebelum/sesudah komputasi lokal)
 *   y = waktu komunikasi (scatter + gather)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

/* ─── Alokasi matriks 1D row-major ─── */
float *allocMatrix(int rows, int cols) {
    float *m = (float *)calloc(rows * cols, sizeof(float));
    if (!m) { fprintf(stderr, "calloc gagal!\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    return m;
}

void initMatrix(float *m, int N) {
    srand(42);
    for (int i = 0; i < N*N; i++) m[i] = (float)rand() / RAND_MAX;
}

/* Hitung baris yang ditangani masing-masing rank */
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

    if (rank == 0) {
        printf("============================================================\n");
        printf("  MPI Perkalian Matriks  |  NP=%d  N=%d\n", np, N);
        printf("============================================================\n");
    }

    /* ─── Rank 0: alokasi dan inisialisasi ─── */
    float *A = NULL, *B = NULL, *C = NULL;
    if (rank == 0) {
        A = allocMatrix(N, N);
        B = allocMatrix(N, N);
        C = allocMatrix(N, N);
        initMatrix(A, N);
        initMatrix(B, N);
    }

    /* ─── Scatter counts dan displacements ─── */
    int *sendCounts = (int *)malloc(np * sizeof(int));
    int *displs     = (int *)malloc(np * sizeof(int));
    for (int r = 0; r < np; r++) {
        sendCounts[r] = rowsForRank(r, np, N) * N;
        displs[r]     = offsetForRank(r, np, N) * N;
    }

    int myRows  = rowsForRank(rank, np, N);
    int myStart = offsetForRank(rank, np, N);

    float *localA = allocMatrix(myRows, N);
    float *localC = allocMatrix(myRows, N);
    float *localB = allocMatrix(N, N);       /* B penuh di setiap rank */

    /* ─── WAKTU KOMUNIKASI: Scatter A + Broadcast B ─── */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_comm_start = MPI_Wtime();

    /* Scatter baris A ke semua rank */
    MPI_Scatterv(A, sendCounts, displs, MPI_FLOAT,
                 localA, myRows * N, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    /* Broadcast seluruh B ke semua rank */
    if (rank == 0) memcpy(localB, B, N*N*sizeof(float));
    MPI_Bcast(localB, N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double t_comm_scatter = MPI_Wtime() - t_comm_start;

    /* ─── WAKTU KOMPUTASI LOKAL ─── */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_compute_start = MPI_Wtime();

    for (int i = 0; i < myRows; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += localA[i*N + k] * localB[k*N + j];
            localC[i*N + j] = sum;
        }
    }

    double t_compute = MPI_Wtime() - t_compute_start;

    /* ─── WAKTU KOMUNIKASI: Gather C ─── */
    double t_gather_start = MPI_Wtime();

    MPI_Gatherv(localC, myRows * N, MPI_FLOAT,
                C, sendCounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    double t_comm_gather = MPI_Wtime() - t_gather_start;
    double t_comm_total  = t_comm_scatter + t_comm_gather;

    /* ─── Reduce waktu ke rank 0 ─── */
    double maxCompute, maxComm;
    MPI_Reduce(&t_compute,    &maxCompute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_total, &maxComm,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double gflops = (2.0 * N * N * N) / (maxCompute * 1e9);

        printf("\n  ┌─────────────────────────────────────────┐\n");
        printf("  │ HASIL PENGUKURAN WAKTU (MPI, NP=%2d)     │\n", np);
        printf("  ├─────────────────────────────────────────┤\n");
        printf("  │ Komputasi (max)   : %10.6f detik    │\n", maxCompute);
        printf("  │ Komunikasi (max)  : %10.6f detik    │\n", maxComm);
        printf("  │   - Scatter+Bcast : %10.6f detik    │\n", t_comm_scatter);
        printf("  │   - Gather        : %10.6f detik    │\n", t_comm_gather);
        printf("  │ GFLOPS            : %10.4f          │\n", gflops);
        printf("  ├─────────────────────────────────────────┤\n");
        printf("  │ FORMAT TABEL (x/y):                     │\n");
        printf("  │   x = %.6f s  y = %.6f s       │\n", maxCompute, maxComm);
        printf("  └─────────────────────────────────────────┘\n");

        /* Verifikasi vs referensi CPU */
        char refFile[64];
        sprintf(refFile, "ref_result_N%d.bin", N);
        FILE *fp = fopen(refFile, "rb");
        if (fp) {
            float *ref = allocMatrix(N, N);
            fread(ref, sizeof(float), N*N, fp); fclose(fp);
            int ok = 1;
            for (int i = 0; i < N*N; i++) {
                if (fabsf(C[i] - ref[i]) > 1e-2f) { ok = 0; break; }
            }
            printf("\n  [VERIFIKASI] %s\n", ok ? "PASS ✓" : "FAIL ✗");
            free(ref);
        } else {
            printf("\n  [VERIFIKASI] File referensi tidak ditemukan.\n");
        }

        printf("============================================================\n");
        free(A); free(B); free(C);
    }

    free(sendCounts); free(displs);
    free(localA); free(localC); free(localB);
    MPI_Finalize();
    return 0;
}
