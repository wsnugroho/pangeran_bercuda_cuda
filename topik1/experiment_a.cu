// experiment_a.cu — Variasi Grid & Block
#include <stdio.h>

__global__ void printThreadInfo(int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        printf("blockIdx=%d threadIdx=%d globalIdx=%d\n",
               blockIdx.x, threadIdx.x, idx);
    }
}

int main() {
    // === VARIASI KONFIGURASI ===
    int configs[][2] = {
        {1, 5},   // 1 block,  5 thread/block  -> 5  total thread
        {3, 5},   // 3 block,  5 thread/block  -> 15 total thread
        {2, 8},   // 2 block,  8 thread/block  -> 16 total thread
        {1, 32},  // 1 block, 32 thread/block  -> 32 total thread
        {4, 1024},
    };
    for (int i = 0; i < 5; i++) {
        int nBlocks = configs[i][0], blockSize = configs[i][1];
        int N = nBlocks * blockSize;
        printf("\n--- Config: %d block x %d threads ---\n", nBlocks, blockSize);
        printThreadInfo<<<nBlocks, blockSize>>>(N);
        cudaDeviceSynchronize();
    }
    return 0;
}
