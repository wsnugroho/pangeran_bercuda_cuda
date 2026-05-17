/*
 * matmul_cuda_basic.cu
 * Perkalian Matriks Bujur Sangkar — CUDA Tanpa Shared Memory
 * Topik 2: Setiap thread menghitung satu elemen C[i][j]
 *
 * Kompilasi: nvcc -O2 -o matmul_cuda_basic matmul_cuda_basic.cu
 * Jalankan : ./matmul_cuda_basic [N] [BLOCK_SIZE]
 *            N          = ukuran matriks (default 512)
 *            BLOCK_SIZE = ukuran block 2D (default 16, max 32)
 *
 * Variasi yang dianjurkan:
 *   ./matmul_cuda_basic 512  8
 *   ./matmul_cuda_basic 512  16
 *   ./matmul_cuda_basic 512  32
 *   ./matmul_cuda_basic 1024 16
 *   ./matmul_cuda_basic 2048 16
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

/* ────────────────────────────────────────────────
   MACRO PEMBANTU
   ──────────────────────────────────────────────── */
#define CUDA_CHECK(call) do {                                          \
    cudaError_t e = (call);                                            \
    if (e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA Error %s:%d — %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(e));            \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

/* ────────────────────────────────────────────────
   KERNEL: Global memory only
   Setiap thread (row, col) menghitung satu elemen C
   ──────────────────────────────────────────────── */
__global__ void matMulBasic(const float *A, const float *B, float *C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // kolom C
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // baris  C

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row*N + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}

/* ────────────────────────────────────────────────
   UTILITAS
   ──────────────────────────────────────────────── */
float *allocHost(int N) {
    float *m = (float *)malloc(N*N*sizeof(float));
    if (!m) { fprintf(stderr, "malloc gagal!\n"); exit(1); }
    return m;
}

void initMatrix(float *m, int N) {
    for (int i = 0; i < N*N; i++) m[i] = (float)rand() / RAND_MAX;
}

int verifyVsRef(const float *C, int N, const char *refFile) {
    FILE *fp = fopen(refFile, "rb");
    if (!fp) { printf("  [SKIP] File referensi '%s' tidak ditemukan.\n", refFile); return -1; }
    float *ref = allocHost(N);
    fread(ref, sizeof(float), N*N, fp); fclose(fp);
    int ok = 1;
    int errCount = 0;
    for (int i = 0; i < N*N; i++) {
        float diff = fabsf(C[i] - ref[i]);
        float tol  = 1e-2f + 1e-4f * fabsf(ref[i]);
        if (diff > tol) {
            if (errCount++ < 5)
                printf("  DIFF[%d]: GPU=%.5f  REF=%.5f\n", i, C[i], ref[i]);
            ok = 0;
        }
    }
    free(ref);
    return ok;
}

/* ────────────────────────────────────────────────
   MAIN
   ──────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    int N          = (argc > 1) ? atoi(argv[1]) : 512;
    int BLOCK_SIZE = (argc > 2) ? atoi(argv[2]) : 16;

    printf("============================================================\n");
    printf("  CUDA Matriks — TANPA Shared Memory\n");
    printf("  N=%d  BLOCK_SIZE=%d\n", N, BLOCK_SIZE);
    printf("============================================================\n");

    /* --- Print info GPU --- */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  VRAM: %.0f MB\n", prop.totalGlobalMem / (1024.0*1024));

    size_t bytes = (size_t)N * N * sizeof(float);

    /* --- Host alloc & init --- */
    float *h_A = allocHost(N);
    float *h_B = allocHost(N);
    float *h_C = allocHost(N);

    /* Seed sekali saja agar A dan B sama persis dengan referensi CPU. */
    srand(42);
    initMatrix(h_A, N);
    initMatrix(h_B, N);
    memset(h_C, 0, bytes);

    /* --- Device alloc --- */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    /* === UKUR WAKTU KOMUNIKASI H→D === */
    cudaEvent_t ev0, ev1, ev2, ev3;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev2));
    CUDA_CHECK(cudaEventCreate(&ev3));

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    /* === KONFIGURASI GRID === */
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE-1)/BLOCK_SIZE,
                 (N + BLOCK_SIZE-1)/BLOCK_SIZE);

    printf("\n  Grid  : (%d, %d)  blocks\n", gridDim.x, gridDim.y);
    printf("  Block : (%d, %d)  threads  -> %d threads/block\n",
           blockDim.x, blockDim.y, BLOCK_SIZE*BLOCK_SIZE);
    printf("  Total threads : %d\n\n", gridDim.x*gridDim.y*BLOCK_SIZE*BLOCK_SIZE);

    /* === JALANKAN KERNEL === */
    CUDA_CHECK(cudaEventRecord(ev2));
    matMulBasic<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(ev3));
    CUDA_CHECK(cudaEventSynchronize(ev3));
    CUDA_CHECK(cudaGetLastError());

    /* === UKUR WAKTU KOMUNIKASI D→H === */
    cudaEvent_t ev4, ev5;
    CUDA_CHECK(cudaEventCreate(&ev4));
    CUDA_CHECK(cudaEventCreate(&ev5));
    CUDA_CHECK(cudaEventRecord(ev4));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev5));
    CUDA_CHECK(cudaEventSynchronize(ev5));

    /* === HITUNG WAKTU === */
    float ms_htod, ms_kernel, ms_dtoh;
    CUDA_CHECK(cudaEventElapsedTime(&ms_htod,   ev0, ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, ev2, ev3));
    CUDA_CHECK(cudaEventElapsedTime(&ms_dtoh,   ev4, ev5));

    double t_comm    = (ms_htod + ms_dtoh) / 1000.0;
    double t_compute = ms_kernel / 1000.0;
    double gflops    = (2.0 * N * N * N) / (t_compute * 1e9);

    printf("  ┌─────────────────────────────────────────┐\n");
    printf("  │ HASIL PENGUKURAN WAKTU                  │\n");
    printf("  ├─────────────────────────────────────────┤\n");
    printf("  │ Transfer H→D      : %10.6f detik    │\n", ms_htod/1000);
    printf("  │ Komputasi (kernel): %10.6f detik    │\n", t_compute);
    printf("  │ Transfer D→H      : %10.6f detik    │\n", ms_dtoh/1000);
    printf("  │ Komunikasi total  : %10.6f detik    │\n", t_comm);
    printf("  │ GFLOPS            : %10.4f          │\n", gflops);
    printf("  ├─────────────────────────────────────────┤\n");
    printf("  │ FORMAT TABEL (x/y):                     │\n");
    printf("  │   x = %.6f s  y = %.6f s       │\n", t_compute, t_comm);
    printf("  └─────────────────────────────────────────┘\n");

    /* === VERIFIKASI === */
    char refFile[64];
    sprintf(refFile, "ref_result_N%d.bin", N);
    int ok = verifyVsRef(h_C, N, refFile);
    if (ok == 1)  printf("\n  [VERIFIKASI] PASS ✓ — Hasil identik dengan CPU\n");
    else if (ok == 0) printf("\n  [VERIFIKASI] FAIL ✗ — Ada perbedaan!\n");

    printf("============================================================\n");

    /* Cleanup */
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
