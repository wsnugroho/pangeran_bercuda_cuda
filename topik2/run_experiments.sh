#!/bin/bash
# run_experiments.sh
# Script otomatis untuk menjalankan semua eksperimen Topik 2
# dan mencatat hasilnya ke file log.
#
# Penggunaan: bash run_experiments.sh
# Output    : experiment_results.log

LOG="experiment_results.log"
echo "==================================================" | tee $LOG
echo "  EKSPERIMEN TOPIK 2: PERKALIAN MATRIKS" | tee -a $LOG
echo "  $(date)" | tee -a $LOG
echo "==================================================" | tee -a $LOG

# ──────────────────────────────────────────
# 1. KOMPILASI
# ──────────────────────────────────────────
echo -e "\n[KOMPILASI]" | tee -a $LOG

echo -e "\n[SETUP CUDA LIBRARY PATH]" | tee -a $LOG
if [ -f ./setup_cuda_libpath.sh ]; then
    source ./setup_cuda_libpath.sh > .cuda_setup.log 2>&1
    cat .cuda_setup.log | tee -a $LOG
fi

gcc  -O2 -o matmul_seq    matmul_sequential.c -lm && echo "  ✓ matmul_seq" | tee -a $LOG
nvcc -O2 -allow-unsupported-compiler -o matmul_basic  matmul_cuda_basic.cu     && echo "  ✓ matmul_basic" | tee -a $LOG
nvcc -O2 -allow-unsupported-compiler -o matmul_shared matmul_cuda_shared.cu    && echo "  ✓ matmul_shared" | tee -a $LOG
nvcc -O2 -allow-unsupported-compiler -o matmul_cublas matmul_cublas.cu -lcublas && echo "  ✓ matmul_cublas" | tee -a $LOG
ldd ./matmul_cublas | grep -E 'cublas|not found' | tee -a $LOG
mpicc -O2 -o matmul_mpi   matmul_mpi.c -lm         && echo "  ✓ matmul_mpi" | tee -a $LOG

# ──────────────────────────────────────────
# 2. UKURAN MATRIKS
# ──────────────────────────────────────────
SIZES="512 1024 2048"
BLOCK_SIZES="8 16 32"

# ──────────────────────────────────────────
# 3. SEQUENTIAL (referensi)
# ──────────────────────────────────────────
echo -e "\n[A] SEQUENTIAL CPU" | tee -a $LOG
for N in $SIZES; do
    echo -e "\n  --- N=$N ---" | tee -a $LOG
    ./matmul_seq $N | tee -a $LOG
done

# ──────────────────────────────────────────
# 4. CUDA TANPA SHARED MEMORY
# ──────────────────────────────────────────
echo -e "\n[B] CUDA TANPA SHARED MEMORY" | tee -a $LOG
for N in $SIZES; do
    for BS in $BLOCK_SIZES; do
        echo -e "\n  --- N=$N  BlockSize=$BS ---" | tee -a $LOG
        ./matmul_basic $N $BS | tee -a $LOG
    done
done

# ──────────────────────────────────────────
# 5. CUDA DENGAN SHARED MEMORY
# ──────────────────────────────────────────
echo -e "\n[C] CUDA DENGAN SHARED MEMORY (TILED)" | tee -a $LOG
for N in $SIZES; do
    for BS in $BLOCK_SIZES; do
        echo -e "\n  --- N=$N  TileSize=$BS ---" | tee -a $LOG
        ./matmul_shared $N $BS | tee -a $LOG
    done
done

# ──────────────────────────────────────────
# 6. cuBLAS
# ──────────────────────────────────────────
echo -e "\n[D] cuBLAS" | tee -a $LOG
for N in $SIZES; do
    echo -e "\n  --- N=$N ---" | tee -a $LOG
    ./matmul_cublas $N | tee -a $LOG
done

# ──────────────────────────────────────────
# 7. MPI
# ──────────────────────────────────────────
echo -e "\n[E] MPI MULTICORE" | tee -a $LOG
for N in $SIZES; do
    for NP in 8 16; do
        echo -e "\n  --- N=$N  NP=$NP ---" | tee -a $LOG
        mpirun --allow-run-as-root -np $NP ./matmul_mpi $N | tee -a $LOG
    done
done

echo -e "\n==================================================" | tee -a $LOG
echo "  SELESAI. Hasil disimpan di: $LOG" | tee -a $LOG
echo "==================================================" | tee -a $LOG
