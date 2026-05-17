/*
 * matmul_sequential.c
 * Perkalian Matriks Bujur Sangkar — Implementasi Sekuensial (CPU)
 * Topik 2: Perbandingan Sequential vs Parallel CUDA
 *
 * Kompilasi: gcc -O2 -o matmul_seq matmul_sequential.c -lm
 * Jalankan : ./matmul_seq [N]   (default N=512)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

/* ─── Alokasi matriks 1D (row-major) ─── */
float *allocMatrix(int N) {
    float *m = (float *)malloc(N * N * sizeof(float));
    if (!m) { fprintf(stderr, "malloc gagal!\n"); exit(1); }
    return m;
}

/* ─── Inisialisasi matriks dengan nilai acak [0,1) ─── */
void initMatrix(float *m, int N) {
    for (int i = 0; i < N * N; i++)
        m[i] = (float)rand() / RAND_MAX;
}

/* ─── Perkalian matriks naive O(N^3) ─── */
void matMulSeq(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
    }
}

/* ─── Verifikasi: bandingkan dua matriks, toleransi 1e-3 ─── */
int verifyResult(const float *C1, const float *C2, int N) {
    for (int i = 0; i < N * N; i++) {
        if (fabsf(C1[i] - C2[i]) > 1e-3f) {
            printf("  MISMATCH pada indeks %d: %.6f vs %.6f\n", i, C1[i], C2[i]);
            return 0;
        }
    }
    return 1;
}

/* ─── Hitung waktu dalam detik ─── */
double getTime(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

int main(int argc, char *argv[]) {
    int N = 512;
    if (argc > 1) N = atoi(argv[1]);

    printf("============================================================\n");
    printf("  Perkalian Matriks Sekuensial (CPU)\n");
    printf("  Ukuran Matriks : N = %d x %d\n", N, N);
    printf("  Memori         : %.2f MB per matriks\n",
           (float)(N*N*sizeof(float)) / (1024*1024));
    printf("============================================================\n");

    srand(42);
    float *A = allocMatrix(N);
    float *B = allocMatrix(N);
    float *C = allocMatrix(N);

    initMatrix(A, N);
    initMatrix(B, N);
    memset(C, 0, N*N*sizeof(float));

    /* Ukur waktu komputasi */
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    matMulSeq(A, B, C, N);
    clock_gettime(CLOCK_MONOTONIC, &t_end);

    double elapsed = getTime(&t_start, &t_end);

    printf("\n  Waktu Komputasi : %.6f detik\n", elapsed);
    printf("  GFLOPS          : %.4f\n",
           (2.0 * N * N * N) / (elapsed * 1e9));
    printf("\n  C[0][0] = %.6f  (sanity check)\n", C[0]);
    printf("============================================================\n");

    /* Simpan hasil ke file untuk verifikasi silang */
    char fname[64];
    sprintf(fname, "ref_result_N%d.bin", N);
    FILE *fp = fopen(fname, "wb");
    if (fp) { fwrite(C, sizeof(float), N*N, fp); fclose(fp); }
    printf("  Referensi hasil disimpan ke: %s\n", fname);
    printf("============================================================\n");

    free(A); free(B); free(C);
    return 0;
}
