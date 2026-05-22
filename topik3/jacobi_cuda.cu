/*
 * jacobi_cuda.cu
 * Jacobi Iteration — CUDA (GPU)
 * Topik 3: Implementasi paralel GPU dengan reduksi max-diff di device
 *
 * Kompilasi: nvcc -O2 -allow-unsupported-compiler -o jacobi_cuda jacobi_cuda.cu
 * Jalankan : ./jacobi_cuda [N] [BLOCK_SIZE]
 *            N          = ukuran sistem (default 512)
 *            BLOCK_SIZE = ukuran block 1D (default 256)
 *
 * Strategi:
 *   - H→D transfer sekali untuk A, b, x.
 *   - Tiap iterasi:
 *       1. Kernel jacobiCompute: tiap thread 1 elemen x_new, simpan diff[i].
 *       2. Kernel reduceMaxPass1: block-level reduction -> d_intermediate.
 *       3. Kernel reduceMaxPass2: 1 block reduce d_intermediate -> d_max_diff.
 *       4. D→H copy 1 float (d_max_diff) untuk cek konvergensi host.
 *       5. Swap pointer d_x_old <-> d_x_new (device).
 *   - D→H transfer akhir: copy x_final ke host.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t e = (call);                                            \
    if (e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA Error %s:%d — %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(e));            \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

/* ─── Kernel: compute x_new dan diff per elemen ─── */
__global__ void jacobiCompute(const float *A, const float *b,
                              const float *x_old, float *x_new,
                              float *diff, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        if (j != i) sum += A[i*N + j] * x_old[j];
    }
    x_new[i] = (b[i] - sum) / A[i*N + i];
    diff[i] = fabsf(x_new[i] - x_old[i]);
}

/* ─── Kernel: block-level reduction max ─── */
__global__ void reduceMaxPass1(const float *in, float *out, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < n) ? in[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

/* ─── Host helpers ─── */
float *allocHost(int n) {
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "malloc gagal!\n"); exit(1); }
    return p;
}

float *allocDevice(int n) {
    float *p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    return p;
}

double getWallTime(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) * 1e-9;
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

int main(int argc, char *argv[]) {
    int N = 512;
    if (argc > 1) N = atoi(argv[1]);
    int blockSize = 256;
    if (argc > 2) blockSize = atoi(argv[2]);

    float eps = 1e-6f;
    int max_iter = 10000;

    printf("============================================================\n");
    printf("  Jacobi Iteration — CUDA (GPU)\n");
    printf("  Ukuran Sistem  : N = %d\n", N);
    printf("  Block Size     : %d\n", blockSize);
    printf("  Epsilon        : %e\n", eps);
    printf("  Max Iterasi    : %d\n", max_iter);
    printf("  Memori A       : %.2f MB\n",
           (float)(N*N*sizeof(float)) / (1024*1024));
    printf("============================================================\n");

    /* ─── Host allocation & init ─── */
    float *h_A = allocHost(N * N);
    float *h_b = allocHost(N);
    float *h_x = allocHost(N);
    initSystem(h_A, h_b, N);
    memset(h_x, 0, N * sizeof(float));

    /* ─── Device allocation ─── */
    float *d_A     = allocDevice(N * N);
    float *d_b     = allocDevice(N);
    float *d_xold  = allocDevice(N);
    float *d_xnew  = allocDevice(N);
    float *d_diff  = allocDevice(N);
    int maxInterSize = (N + blockSize - 1) / blockSize;
    float *d_inter1 = allocDevice(maxInterSize);
    float *d_inter2 = allocDevice(maxInterSize);

    /* ─── Timing events ─── */
    cudaEvent_t ev_start, ev_stop, ev_h2d_start, ev_h2d_stop, ev_d2h_start, ev_d2h_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_stop));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));

    /* ─── H→D transfer (sekali) ─── */
    CUDA_CHECK(cudaEventRecord(ev_h2d_start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_xold, 0, N*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_xnew, 0, N*sizeof(float)));
    CUDA_CHECK(cudaEventRecord(ev_h2d_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_h2d_stop));

    int numBlocks = (N + blockSize - 1) / blockSize;

    /* ─── Jacobi loop di device ─── */
    CUDA_CHECK(cudaEventRecord(ev_start));

    int iter = 0;
    float h_max_diff = eps + 1.0f;
    double t_sync_check = 0.0;

    while (iter < max_iter && h_max_diff >= eps) {
        jacobiCompute<<<numBlocks, blockSize>>>(d_A, d_b, d_xold, d_xnew, d_diff, N);
        CUDA_CHECK(cudaGetLastError());

        /* Iterative reduction: robust untuk semua ukuran */
        int currentN = N;
        float *d_src = d_diff;
        float *d_dst = d_inter1;
        while (currentN > 1) {
            int blocks = (currentN + blockSize - 1) / blockSize;
            reduceMaxPass1<<<blocks, blockSize, blockSize*sizeof(float)>>>(d_src, d_dst, currentN);
            CUDA_CHECK(cudaGetLastError());
            float *tmp = d_src; d_src = d_dst; d_dst = tmp;
            currentN = blocks;
        }

        struct timespec t_sync_start, t_sync_end;
        clock_gettime(CLOCK_MONOTONIC, &t_sync_start);
        CUDA_CHECK(cudaMemcpy(&h_max_diff, d_src, sizeof(float), cudaMemcpyDeviceToHost));
        clock_gettime(CLOCK_MONOTONIC, &t_sync_end);
        t_sync_check += getWallTime(&t_sync_start, &t_sync_end);

        /* swap pointers */
        float *tmp = d_xold; d_xold = d_xnew; d_xnew = tmp;
        iter++;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    /* ─── D→H transfer akhir ─── */
    CUDA_CHECK(cudaEventRecord(ev_d2h_start));
    CUDA_CHECK(cudaMemcpy(h_x, d_xold, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_stop));

    float ms_loop = 0.0f, ms_h2d = 0.0f, ms_d2h = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_loop, ev_start, ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ev_h2d_start, ev_h2d_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ev_d2h_start, ev_d2h_stop));

    double t_loop   = ms_loop   / 1000.0;
    double t_h2d    = ms_h2d    / 1000.0;
    double t_d2h    = ms_d2h    / 1000.0;
    double t_kernel = t_loop - t_sync_check;
    if (t_kernel < 0.0) t_kernel = 0.0;
    double t_total  = t_loop + t_h2d + t_d2h;

    printf("\n  Waktu Compute GPU : %.6f detik\n", t_kernel);
    printf("  Sync convergence : %.6f detik\n", t_sync_check);
    printf("  Transfer H→D    : %.6f detik\n", t_h2d);
    printf("  Transfer D→H akhir: %.6f detik\n", t_d2h);
    printf("  Waktu Total     : %.6f detik\n", t_total);
    printf("  Iterasi         : %d\n", iter);
    printf("  GFLOPS (compute): %.4f\n",
           (t_kernel > 0.0) ? (2.0 * N * N * iter) / (t_kernel * 1e9) : 0.0);
    printf("\n  x[0] = %.6f  (sanity check)\n", h_x[0]);
    printf("============================================================\n");

    /* ─── Verifikasi vs referensi CPU ─── */
    char refFile[64];
    sprintf(refFile, "ref_x_N%d.bin", N);
    FILE *fp = fopen(refFile, "rb");
    if (fp) {
        float *ref = allocHost(N);
        if (fread(ref, sizeof(float), N, fp) != (size_t)N) {
            printf("  [VERIFIKASI] Gagal membaca file referensi.\n");
            free(ref);
            fclose(fp);
            printf("============================================================\n");
            cudaFree(d_A); cudaFree(d_b); cudaFree(d_xold); cudaFree(d_xnew);
            cudaFree(d_diff); cudaFree(d_inter1); cudaFree(d_inter2);
            free(h_A); free(h_b); free(h_x);
            cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
            cudaEventDestroy(ev_h2d_start); cudaEventDestroy(ev_h2d_stop);
            cudaEventDestroy(ev_d2h_start); cudaEventDestroy(ev_d2h_stop);
            return 1;
        }
        fclose(fp);
        int ok = 1;
        for (int i = 0; i < N; i++) {
            float diff = fabsf(h_x[i] - ref[i]);
            float tol  = 1e-2f + 1e-4f * fabsf(ref[i]);
            if (diff > tol) { ok = 0; break; }
        }
        printf("  [VERIFIKASI] %s\n", ok ? "PASS ✓" : "FAIL ✗");
        free(ref);
    } else {
        printf("  [VERIFIKASI] File referensi tidak ditemukan.\n");
    }
    printf("============================================================\n");

    /* Cleanup */
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_xold); cudaFree(d_xnew);
    cudaFree(d_diff); cudaFree(d_inter1); cudaFree(d_inter2);
    free(h_A); free(h_b); free(h_x);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    cudaEventDestroy(ev_h2d_start); cudaEventDestroy(ev_h2d_stop);
    cudaEventDestroy(ev_d2h_start); cudaEventDestroy(ev_d2h_stop);
    return 0;
}
