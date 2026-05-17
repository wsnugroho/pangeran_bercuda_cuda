/*
 * matmul_cuda_shared.cu
 * Perkalian Matriks Bujur Sangkar — CUDA DENGAN Shared Memory (Tiled)
 * Topik 2: Optimisasi menggunakan tile-based shared memory
 *
 * Kompilasi: nvcc -O2 -o matmul_cuda_shared matmul_cuda_shared.cu
 * Jalankan : ./matmul_cuda_shared [N] [TILE_SIZE]
 *            TILE_SIZE harus sama dengan BLOCK_SIZE (default 16)
 *
 * Variasi:
 *   ./matmul_cuda_shared 512  8
 *   ./matmul_cuda_shared 512  16
 *   ./matmul_cuda_shared 512  32
 *   ./matmul_cuda_shared 1024 16
 *   ./matmul_cuda_shared 2048 16
 *
 * Strategi Tiling:
 *   Matriks A dan B dipotong jadi "tile" berukuran TILE x TILE.
 *   Setiap block memuat satu tile A dan satu tile B ke shared memory,
 *   lalu menghitung partial dot-product. Ini drastis mengurangi akses
 *   ke global memory (bandwidth bottleneck).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_TILE 32   /* shared mem max tile yang aman: 32x32x4x2 = 8KB < 48KB */

#define CUDA_CHECK(call) do {                                          \
    cudaError_t e = (call);                                            \
    if (e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA Error %s:%d — %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(e));            \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

/* ────────────────────────────────────────────────
   KERNEL: Tiled Shared Memory MatMul
   Template parameter T = tile size (compile-time constant)
   ──────────────────────────────────────────────── */
template <int T>
__global__ void matMulShared(const float *A, const float *B, float *C, int N) {
    /* Shared memory tiles — alokasi statis */
    __shared__ float tileA[T][T];
    __shared__ float tileB[T][T];

    int tx = threadIdx.x,  ty = threadIdx.y;
    int row = blockIdx.y * T + ty;   /* baris C yang dihitung thread ini */
    int col = blockIdx.x * T + tx;   /* kolom C yang dihitung thread ini */

    float sum = 0.0f;

    /* Loop atas semua tile sepanjang dimensi K */
    int numTiles = (N + T - 1) / T;
    for (int t = 0; t < numTiles; t++) {
        /* Muat tile A[row][t*T + tx] ke shared memory */
        int aCol = t * T + tx;
        tileA[ty][tx] = (row < N && aCol < N) ? A[row*N + aCol] : 0.0f;

        /* Muat tile B[t*T + ty][col] ke shared memory */
        int bRow = t * T + ty;
        tileB[ty][tx] = (bRow < N && col < N) ? B[bRow*N + col] : 0.0f;

        /* Sinkronisasi: pastikan semua thread sudah selesai memuat */
        __syncthreads();

        /* Komputasi partial dot-product dari tile ini */
        #pragma unroll
        for (int k = 0; k < T; k++)
            sum += tileA[ty][k] * tileB[k][tx];

        /* Sinkronisasi: jangan muat tile berikutnya sebelum semua selesai */
        __syncthreads();
    }

    /* Tulis hasil ke global memory */
    if (row < N && col < N)
        C[row*N + col] = sum;
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
    srand(42);
    for (int i = 0; i < N*N; i++) m[i] = (float)rand() / RAND_MAX;
}

int verifyVsRef(const float *C, int N, const char *refFile) {
    FILE *fp = fopen(refFile, "rb");
    if (!fp) { printf("  [SKIP] File referensi '%s' tidak ditemukan.\n", refFile); return -1; }
    float *ref = allocHost(N);
    fread(ref, sizeof(float), N*N, fp); fclose(fp);
    int ok = 1; int errCount = 0;
    for (int i = 0; i < N*N; i++) {
        if (fabsf(C[i] - ref[i]) > 1e-2f) {
            if (errCount++ < 5)
                printf("  DIFF[%d]: GPU=%.5f  REF=%.5f\n", i, C[i], ref[i]);
            ok = 0;
        }
    }
    free(ref);
    return ok;
}

/* Dispatcher: pilih tile size yang tepat saat runtime */
void launchKernel(dim3 grid, dim3 block, const float *dA, const float *dB,
                  float *dC, int N, int T) {
    switch (T) {
        case  8: matMulShared< 8><<<grid, block>>>(dA, dB, dC, N); break;
        case 16: matMulShared<16><<<grid, block>>>(dA, dB, dC, N); break;
        case 32: matMulShared<32><<<grid, block>>>(dA, dB, dC, N); break;
        default:
            fprintf(stderr, "TILE_SIZE %d tidak didukung (gunakan 8,16,32)\n", T);
            exit(1);
    }
}

/* ────────────────────────────────────────────────
   MAIN
   ──────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    int N         = (argc > 1) ? atoi(argv[1]) : 512;
    int TILE_SIZE = (argc > 2) ? atoi(argv[2]) : 16;

    printf("============================================================\n");
    printf("  CUDA Matriks — DENGAN Shared Memory (Tiled)\n");
    printf("  N=%d  TILE_SIZE=%d\n", N, TILE_SIZE);
    printf("============================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU          : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Shared Mem   : %zu KB per block\n", prop.sharedMemPerBlock / 1024);
    printf("  Shared dipakai: %d KB (tile A + tile B)\n",
           2 * TILE_SIZE * TILE_SIZE * (int)sizeof(float) / 1024);

    size_t bytes = (size_t)N * N * sizeof(float);

    /* Host */
    float *h_A = allocHost(N);
    float *h_B = allocHost(N);
    float *h_C = allocHost(N);
    initMatrix(h_A, N);
    initMatrix(h_B, N);
    memset(h_C, 0, bytes);

    /* Device */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    /* Timing events */
    cudaEvent_t ev[6];
    for (int i = 0; i < 6; i++) CUDA_CHECK(cudaEventCreate(&ev[i]));

    /* Transfer H→D */
    CUDA_CHECK(cudaEventRecord(ev[0]));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev[1]));
    CUDA_CHECK(cudaEventSynchronize(ev[1]));

    /* Konfigurasi */
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE-1)/TILE_SIZE,
                 (N + TILE_SIZE-1)/TILE_SIZE);

    printf("\n  Grid  : (%d, %d) blocks\n", gridDim.x, gridDim.y);
    printf("  Block : (%d, %d) threads  -> %d threads/block\n",
           blockDim.x, blockDim.y, TILE_SIZE*TILE_SIZE);
    printf("  Total threads: %d\n\n",
           gridDim.x * gridDim.y * TILE_SIZE * TILE_SIZE);

    /* Kernel */
    CUDA_CHECK(cudaEventRecord(ev[2]));
    launchKernel(gridDim, blockDim, d_A, d_B, d_C, N, TILE_SIZE);
    CUDA_CHECK(cudaEventRecord(ev[3]));
    CUDA_CHECK(cudaEventSynchronize(ev[3]));
    CUDA_CHECK(cudaGetLastError());

    /* Transfer D→H */
    CUDA_CHECK(cudaEventRecord(ev[4]));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev[5]));
    CUDA_CHECK(cudaEventSynchronize(ev[5]));

    /* Hitung waktu */
    float ms_htod, ms_kernel, ms_dtoh;
    CUDA_CHECK(cudaEventElapsedTime(&ms_htod,   ev[0], ev[1]));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, ev[2], ev[3]));
    CUDA_CHECK(cudaEventElapsedTime(&ms_dtoh,   ev[4], ev[5]));

    double t_comm    = (ms_htod + ms_dtoh) / 1000.0;
    double t_compute = ms_kernel / 1000.0;
    double gflops    = (2.0 * N * N * N) / (t_compute * 1e9);

    printf("  ┌─────────────────────────────────────────┐\n");
    printf("  │ HASIL PENGUKURAN WAKTU (SHARED MEM)     │\n");
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

    /* Verifikasi */
    char refFile[64];
    sprintf(refFile, "ref_result_N%d.bin", N);
    int ok = verifyVsRef(h_C, N, refFile);
    if (ok == 1)  printf("\n  [VERIFIKASI] PASS ✓ — Hasil identik dengan CPU\n");
    else if (ok == 0) printf("\n  [VERIFIKASI] FAIL ✗ — Ada perbedaan!\n");

    printf("============================================================\n");

    for (int i = 0; i < 6; i++) cudaEventDestroy(ev[i]);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
