// increment_experiment.cu — Eksperimen variasi N & nBlocks
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

__global__ void incrementOnDevice(float *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) a[idx] = a[idx] + 1.f;  // Guard penting!
}

// Fungsi untuk menjalankan satu eksperimen dengan N dan blockSize tertentu
void runExperiment(int N, int blockSize) {
    float *a_h = (float*)malloc(N * sizeof(float));
    float *b_h = (float*)malloc(N * sizeof(float));
    float *a_d;
    cudaMalloc((void**)&a_d, N * sizeof(float));

    // Init data
    for (int i = 0; i < N; i++) a_h[i] = (float)i;
    cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice);

    // === CEILING DIVISION untuk handle N sembarang ===
    int nBlocks = (N + blockSize - 1) / blockSize; // Ekuivalen dengan ceil(N/blockSize)

    printf("[N=%d, blockSize=%d] -> nBlocks=%d, total_threads=%d\n",
           N, blockSize, nBlocks, nBlocks * blockSize);

    // Cek batas hardware (blockSize max 1024)
    if (blockSize > 1024) {
        printf("  ERROR: blockSize %d melebihi batas GPU (max 1024)!\n", blockSize);
        free(a_h); free(b_h); cudaFree(a_d); return;
    }

    // Jalankan kernel
    incrementOnDevice<<<nBlocks, blockSize>>>(a_d, N);
    cudaDeviceSynchronize();

    // Cek error CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA Error: %s\n", cudaGetErrorString(err));
        free(a_h); free(b_h); cudaFree(a_d); return;
    }

    // Retrieve dan verifikasi
    cudaMemcpy(b_h, a_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) a_h[i] += 1.f;
    int ok = 1;
    for (int i = 0; i < N; i++) if (a_h[i] != b_h[i]) { ok = 0; break; }
    printf("  Hasil: %s\n", ok ? "BENAR ✓" : "SALAH ✗");

    free(a_h); free(b_h); cudaFree(a_d);
}

int main(void) {
    printf("=== Eksperimen Variasi N dan blockSize ===\n");
    // Variasi N kecil
    runExperiment(10, 4);   // N tidak habis dibagi blockSize
    runExperiment(10, 10);  // N tepat sama dengan blockSize
    runExperiment(10, 32);  // blockSize > N (ada thread idle)
    // Variasi N besar
    runExperiment(1000, 32);
    runExperiment(100000, 256);
    // Uji batas hardware
    runExperiment(100, 1024); // max blockSize
    runExperiment(100, 1025); // melebihi batas -> error
    return 0;
}
