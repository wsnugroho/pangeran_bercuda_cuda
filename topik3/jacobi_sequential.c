/*
 * jacobi_sequential.c
 * Jacobi Iteration — Sequential CPU (Ground Truth)
 * Topik 3: Solusi Sistem Persamaan Linear dengan Jacobi
 *
 * Kompilasi: gcc -O2 -o jacobi_seq jacobi_sequential.c -lm
 * Jalankan : ./jacobi_seq [N]   (default N=512)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* ─── Alokasi matriks 1D (row-major) ─── */
float *allocMatrix(int N) {
    float *m = (float *)malloc(N * N * sizeof(float));
    if (!m) { fprintf(stderr, "malloc gagal!\n"); exit(1); }
    return m;
}

float *allocVector(int N) {
    float *v = (float *)malloc(N * sizeof(float));
    if (!v) { fprintf(stderr, "malloc gagal!\n"); exit(1); }
    return v;
}

/* ─── Inisialisasi A strictly diagonally dominant ───
   Agar Jacobi pasti konvergen. */
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
        A[i*N + i] = 1.0f + row_sum;  /* dominan diagonal */
        b[i] = (float)rand() / RAND_MAX;
    }
}

/* ─── Jacobi iteration ───
   *iter_out = jumlah iterasi yang dilakukan
   Return 0 jika hasil akhir di buf0, 1 jika di buf1 ─── */
int jacobiSeq(const float *A, const float *b, float *buf0, float *buf1,
              int N, float eps, int max_iter, int *iter_out) {
    float *src = buf0;
    float *dst = buf1;
    for (int iter = 0; iter < max_iter; iter++) {
        float max_diff = 0.0f;
        for (int i = 0; i < N; i++) {
            float sum = 0.0f;
            for (int j = 0; j < i; j++) sum += A[i*N + j] * src[j];
            for (int j = i + 1; j < N; j++) sum += A[i*N + j] * src[j];
            dst[i] = (b[i] - sum) / A[i*N + i];
            float diff = fabsf(dst[i] - src[i]);
            if (diff > max_diff) max_diff = diff;
        }
        float *tmp = src; src = dst; dst = tmp;
        if (max_diff < eps) {
            *iter_out = iter + 1;
            return (src == buf0) ? 0 : 1;
        }
    }
    *iter_out = max_iter;
    return (src == buf0) ? 0 : 1;
}

/* ─── Waktu detik ─── */
double getTime(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

int main(int argc, char *argv[]) {
    int N = 512;
    if (argc > 1) N = atoi(argv[1]);

    float eps = 1e-6f;
    int max_iter = 10000;

    printf("============================================================\n");
    printf("  Jacobi Iteration — Sequential (CPU)\n");
    printf("  Ukuran Sistem  : N = %d\n", N);
    printf("  Epsilon        : %e\n", eps);
    printf("  Max Iterasi    : %d\n", max_iter);
    printf("  Memori A       : %.2f MB\n",
           (float)(N*N*sizeof(float)) / (1024*1024));
    printf("============================================================\n");

    float *A = allocMatrix(N);
    float *b = allocVector(N);
    float *x0 = allocVector(N);
    float *x1 = allocVector(N);

    initSystem(A, b, N);
    memset(x0, 0, N * sizeof(float));
    memset(x1, 0, N * sizeof(float));

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    int iter;
    int buf_idx = jacobiSeq(A, b, x0, x1, N, eps, max_iter, &iter);
    clock_gettime(CLOCK_MONOTONIC, &t_end);

    double elapsed = getTime(&t_start, &t_end);
    float *x_final = (buf_idx == 0) ? x0 : x1;

    printf("\n  Waktu Komputasi : %.6f detik\n", elapsed);
    printf("  Iterasi         : %d\n", iter);
    printf("  GFLOPS          : %.4f\n",
           (2.0 * N * N * iter) / (elapsed * 1e9));
    printf("\n  x[0] = %.6f  (sanity check)\n", x_final[0]);
    printf("============================================================\n");

    char fname[64];
    sprintf(fname, "ref_x_N%d.bin", N);
    FILE *fp = fopen(fname, "wb");
    if (fp) { fwrite(x_final, sizeof(float), N, fp); fclose(fp); }
    printf("  Referensi x disimpan ke: %s\n", fname);
    printf("============================================================\n");

    free(A); free(b); free(x0); free(x1);
    return 0;
}
