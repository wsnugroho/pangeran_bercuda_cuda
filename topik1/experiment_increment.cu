// experiment_increment_v2.cu
// Eksperimen variasi N, gridDim, dan blockDim untuk operasi increment array.
// Fitur baru:
// 1. Bisa memilih GPU dengan argumen --device <id>
// 2. Bisa menampilkan daftar GPU dengan --list-devices
// 3. Mengecek batas maksimum thread per block dari GPU yang dipilih
// 4. Menghitung nBlocks, total thread, dan idle thread
// 5. Mengukur waktu kernel menggunakan CUDA event
// 6. Verifikasi hasil BENAR/SALAH
// 7. Output ringkas dalam bentuk tabel agar mudah dibandingkan antar-GPU
//
// Compile:
//   nvcc experiment_increment_v2.cu -o experiment_increment_v2
//
// Contoh run:
//   ./experiment_increment_v2 --list-devices
//   ./experiment_increment_v2 --device 0
//   ./experiment_increment_v2 --device 1
//
// Alternatif run dengan environment variable:
//   CUDA_VISIBLE_DEVICES=0 ./experiment_increment_v2
//   CUDA_VISIBLE_DEVICES=1 ./experiment_increment_v2

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call) do {                                                     \
    cudaError_t err__ = (call);                                                    \
    if (err__ != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                cudaGetErrorString(err__));                                        \
        exit(EXIT_FAILURE);                                                        \
    }                                                                              \
} while (0)

__global__ void incrementOnDevice(float *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = a[idx] + 1.0f;
    }
}

static void listDevices(void) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));

    printf("Daftar GPU CUDA yang terdeteksi:\n");
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("  GPU %d: %s | compute capability %d.%d | maxThreadsPerBlock=%d\n",
               i, prop.name, prop.major, prop.minor, prop.maxThreadsPerBlock);
    }
}

static int parseDeviceId(int argc, char **argv) {
    int deviceId = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--list-devices") == 0) {
            listDevices();
            exit(EXIT_SUCCESS);
        }
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            deviceId = atoi(argv[i + 1]);
            i++;
        }
    }
    return deviceId;
}

static int verifyResult(const float *result, int N) {
    for (int i = 0; i < N; i++) {
        float expected = (float)i + 1.0f;
        if (fabsf(result[i] - expected) > 1e-5f) {
            return 0;
        }
    }
    return 1;
}

static void runExperiment(int N, int blockSize, const cudaDeviceProp *prop) {
    if (blockSize <= 0) {
        printf("%-10d %-10d %-10s %-14s %-12s %-14s %-12s\n",
               N, blockSize, "SKIP", "-", "-", "-", "block<=0");
        return;
    }

    if (blockSize > prop->maxThreadsPerBlock) {
        printf("%-10d %-10d %-10s %-14s %-12s %-14s %-12s\n",
               N, blockSize, "SKIP", "-", "-", "-", "invalid");
        return;
    }

    int nBlocks = (N + blockSize - 1) / blockSize;
    int totalThreads = nBlocks * blockSize;
    int idleThreads = totalThreads - N;

    float *a_h = (float*)malloc((size_t)N * sizeof(float));
    float *b_h = (float*)malloc((size_t)N * sizeof(float));
    if (a_h == NULL || b_h == NULL) {
        fprintf(stderr, "Gagal alokasi memori host untuk N=%d.\n", N);
        free(a_h);
        free(b_h);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        a_h[i] = (float)i;
    }

    float *a_d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&a_d, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(a_d, a_h, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up singkat agar timing lebih stabil.
    incrementOnDevice<<<nBlocks, blockSize>>>(a_d, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset data setelah warm-up.
    CUDA_CHECK(cudaMemcpy(a_d, a_h, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    incrementOnDevice<<<nBlocks, blockSize>>>(a_d, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(b_h, a_d, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    int ok = verifyResult(b_h, N);

    printf("%-10d %-10d %-10d %-14d %-12d %-14.6f %-12s\n",
           N, blockSize, nBlocks, totalThreads, idleThreads,
           milliseconds, ok ? "BENAR" : "SALAH");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(a_d));
    free(a_h);
    free(b_h);
}

int main(int argc, char **argv) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Tidak ada GPU CUDA yang terdeteksi.\n");
        return EXIT_FAILURE;
    }

    int deviceId = parseDeviceId(argc, argv);
    if (deviceId < 0 || deviceId >= deviceCount) {
        fprintf(stderr, "Device ID %d tidak valid. Jumlah GPU terdeteksi: %d\n",
                deviceId, deviceCount);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(deviceId));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    printf("=== Experiment Increment v2: Variasi N dan blockSize ===\n");
    printf("GPU aktif: %d - %s\n", deviceId, prop.name);
    printf("Max threads per block GPU ini: %d\n\n", prop.maxThreadsPerBlock);

    printf("%-10s %-10s %-10s %-14s %-12s %-14s %-12s\n",
           "N", "block", "nBlocks", "total_threads", "idle", "kernel_ms", "hasil");
    printf("------------------------------------------------------------------------------------------\n");

    // Variasi bawaan dari eksperimen awal.
    runExperiment(10, 4, &prop);
    runExperiment(10, 10, &prop);
    runExperiment(10, 32, &prop);
    runExperiment(1000, 32, &prop);
    runExperiment(100000, 256, &prop);
    runExperiment(100, 1024, &prop);
    runExperiment(100, 1025, &prop);

    // Variasi tambahan untuk membandingkan performa GTX 1080 Ti vs RTX 3080.
    runExperiment(100000, 32, &prop);
    runExperiment(100000, 64, &prop);
    runExperiment(100000, 128, &prop);
    runExperiment(100000, 512, &prop);
    runExperiment(1000000, 128, &prop);
    runExperiment(1000000, 256, &prop);
    runExperiment(1000000, 512, &prop);
    runExperiment(1000000, 1024, &prop);

    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
