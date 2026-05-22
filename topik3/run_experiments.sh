#!/bin/bash
# rerun_experiments_fixed.sh
# Re-run experiments N=512,1024,2048,4096 setelah bug fix
# Output: experiment_results_fixed.log

LOG="experiment_results_fixed.log"
echo "==================================================" | tee $LOG
echo "  EKSPERIMEN TOPIK 3: JACOBI ITERATION (FIXED)" | tee -a $LOG
echo "  $(date)" | tee -a $LOG
echo "==================================================" | tee -a $LOG

# Kompilasi
echo -e "\n[KOMPILASI]" | tee -a $LOG
gcc  -O2 -o jacobi_seq   jacobi_sequential.c -lm && echo "  ✓ jacobi_seq" | tee -a $LOG
mpicc -O2 -o jacobi_mpi  jacobi_mpi.c -lm         && echo "  ✓ jacobi_mpi" | tee -a $LOG
nvcc -O2 -allow-unsupported-compiler -o jacobi_cuda jacobi_cuda.cu && echo "  ✓ jacobi_cuda" | tee -a $LOG

SIZES="512 1024 2048 4096"

# Sequential
echo -e "\n[A] SEQUENTIAL CPU" | tee -a $LOG
for N in $SIZES; do
    echo -e "\n  --- N=$N ---" | tee -a $LOG
    ./jacobi_seq $N | tee -a $LOG
done

# CUDA (BS=128 optimal)
echo -e "\n[B] CUDA GPU (BlockSize=128)" | tee -a $LOG
for N in $SIZES; do
    echo -e "\n  --- N=$N ---" | tee -a $LOG
    ./jacobi_cuda $N 128 | tee -a $LOG
done

# MPI
echo -e "\n[C] MPI MULTICORE" | tee -a $LOG
for N in $SIZES; do
    for NP in 4 8; do
        echo -e "\n  --- N=$N  NP=$NP ---" | tee -a $LOG
        mpirun --allow-run-as-root -np $NP ./jacobi_mpi $N | tee -a $LOG
    done
done

echo -e "\n==================================================" | tee -a $LOG
echo "  SELESAI. Hasil disimpan di: $LOG" | tee -a $LOG
echo "==================================================" | tee -a $LOG
