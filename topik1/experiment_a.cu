// experiment_a_v2.cu
// Eksperimen variasi gridDim dan blockDim pada CUDA.
// Fitur baru:
// 1. Bisa memilih GPU dengan argumen --device <id>
// 2. Bisa menampilkan daftar GPU dengan --list-devices
// 3. Mengecek batas maksimum thread per block dari GPU yang dipilih
// 4. Menampilkan ringkasan konfigurasi, total thread, dan idle thread
// 5. Mengecek error kernel dan sinkronisasi
//
// Compile:
//   nvcc experiment_a_v2.cu -o experiment_a_v2
//
// Contoh run:
//   ./experiment_a_v2 --list-devices
//   ./experiment_a_v2 --device 0
//   ./experiment_a_v2 --device 1
//
// Alternatif run dengan environment variable:
//   CUDA_VISIBLE_DEVICES=0 ./experiment_a_v2
//   CUDA_VISIBLE_DEVICES=1 ./experiment_a_v2

#include <cuda_runtime.h>
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

__global__ void printThreadInfo(int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        printf("blockIdx=%d threadIdx=%d globalIdx=%d\n",
               blockIdx.x, threadIdx.x, idx);
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
    printf("=== Experiment A v2: Variasi Grid dan Block ===\n");
    printf("GPU aktif: %d - %s\n", deviceId, prop.name);
    printf("Max threads per block GPU ini: %d\n", prop.maxThreadsPerBlock);
    printf("Catatan: urutan printf dari thread CUDA bisa berbeda-beda karena eksekusi paralel.\n");

    // Variasi konfigurasi: {jumlah block, thread per block}
    int configs[][2] = {
        {1, 5},
        {3, 5},
        {2, 8},
        {1, 32},
        {4, 1024}
    };

    int totalConfigs = (int)(sizeof(configs) / sizeof(configs[0]));

    for (int i = 0; i < totalConfigs; i++) {
        int nBlocks = configs[i][0];
        int blockSize = configs[i][1];
        int N = nBlocks * blockSize;

        printf("\n--- Config %d: gridDim.x=%d, blockDim.x=%d ---\n",
               i + 1, nBlocks, blockSize);
        printf("N=%d, total_threads=%d, idle_threads=%d\n",
               N, nBlocks * blockSize, nBlocks * blockSize - N);

        if (blockSize > prop.maxThreadsPerBlock) {
            printf("SKIP: blockDim.x=%d melebihi batas GPU %s yaitu %d.\n",
                   blockSize, prop.name, prop.maxThreadsPerBlock);
            continue;
        }

        printThreadInfo<<<nBlocks, blockSize>>>(N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
