/*
 * matmul_cublas.cu
 * Perkalian Matriks Bujur Sangkar — cuBLAS (NVIDIA Optimized Library)
 * Topik 2: Menggunakan cublasSgemm untuk perbandingan performa
 *
 * Kompilasi: nvcc -O2 -o matmul_cublas matmul_cublas.cu -lcublas
 * Jalankan : ./matmul_cublas [N]
 *
 * CATATAN: cuBLAS menggunakan kolom-mayor (column-major) secara internal.
 *          Kita memanfaatkan trik: C = A*B dalam row-major
 *          ekuivalen dengan C^T = B^T * A^T dalam column-major,
 *          sehingga kita memanggil cublasSgemm(B, A) tanpa transpose.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t e = (call);                                            \
    if (e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA Error %s:%d — %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(e));            \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

#define CUBLAS_CHECK(call) do {                                        \
    cublasStatus_t s = (call);                                         \
    if (s != CUBLAS_STATUS_SUCCESS) {                                  \
        fprintf(stderr, "cuBLAS Error %s:%d — status %d\n",           \
                __FILE__, __LINE__, (int)s);                           \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

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
    int ok = 1; int errCount = 0;
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

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;

    printf("============================================================\n");
    printf("  cuBLAS — Perkalian Matriks Optimal\n");
    printf("  N = %d x %d\n", N, N);
    printf("============================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    size_t bytes = (size_t)N * N * sizeof(float);

    /* Host */
    float *h_A = allocHost(N);
    float *h_B = allocHost(N);
    float *h_C = allocHost(N);

    /* Seed sekali saja agar A dan B sama persis dengan referensi CPU. */
    srand(42);
    initMatrix(h_A, N);
    initMatrix(h_B, N);
    memset(h_C, 0, bytes);

    /* Device */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    /* cuBLAS handle */
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    /* Timing events */
    cudaEvent_t ev[6];
    for (int i = 0; i < 6; i++) CUDA_CHECK(cudaEventCreate(&ev[i]));

    /* Transfer H→D */
    CUDA_CHECK(cudaEventRecord(ev[0]));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev[1]));
    CUDA_CHECK(cudaEventSynchronize(ev[1]));

    /* ─────────────────────────────────────────────
       cublasSgemm:  C = alpha * op(A) * op(B) + beta * C
       Row-major C=A*B  ≡  Column-major C^T = B^T * A^T
       Kita panggil: C^T = 1*B*A + 0*C^T
         → handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
           &alpha, d_B, N, d_A, N, &beta, d_C, N
       ───────────────────────────────────────────── */
    const float alpha = 1.0f, beta = 0.0f;

    CUDA_CHECK(cudaEventRecord(ev[2]));
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_B, N,   /* B  (col-major) */
                             d_A, N,   /* A  (col-major) */
                             &beta,
                             d_C, N)); /* C  (col-major) */
    CUDA_CHECK(cudaEventRecord(ev[3]));
    CUDA_CHECK(cudaEventSynchronize(ev[3]));

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

    printf("\n  ┌─────────────────────────────────────────┐\n");
    printf("  │ HASIL PENGUKURAN WAKTU (cuBLAS)         │\n");
    printf("  ├─────────────────────────────────────────┤\n");
    printf("  │ Transfer H→D      : %10.6f detik    │\n", ms_htod/1000);
    printf("  │ Komputasi cuBLAS  : %10.6f detik    │\n", t_compute);
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
    else if (ok == 0) printf("\n  [VERIFIKASI] FAIL ✗\n");

    printf("============================================================\n");

    cublasDestroy(handle);
    for (int i = 0; i < 6; i++) cudaEventDestroy(ev[i]);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
