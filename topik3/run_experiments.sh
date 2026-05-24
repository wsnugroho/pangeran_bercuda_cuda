#!/bin/bash
# run_experiments.sh
# Runner lokal untuk eksperimen Topik 3 Jacobi.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/experiment_results.log"

N_LIST="512 1024 2048"
MPI_NP_LIST="4 8 16"
CUDA_BLOCK_LIST="128 256 512"

cd "$SCRIPT_DIR"

echo "==================================================" | tee "$LOG"
echo "  EKSPERIMEN TOPIK 3: JACOBI ITERATION" | tee -a "$LOG"
echo "  $(date)" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

echo -e "\n[ENVIRONMENT]" | tee -a "$LOG"
if command -v hostname >/dev/null 2>&1; then
  echo "  Hostname: $(hostname)" | tee -a "$LOG"
fi
if command -v nproc >/dev/null 2>&1; then
  echo "  CPU(s): $(nproc)" | tee -a "$LOG"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU: /' | tee -a "$LOG"
else
  echo "  GPU: nvidia-smi tidak ditemukan" | tee -a "$LOG"
fi

echo -e "\n[SETUP CUDA LIBRARY PATH]" | tee -a "$LOG"
if [ -f "$SCRIPT_DIR/../topik2/setup_cuda_libpath.sh" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/../topik2/setup_cuda_libpath.sh" >.cuda_setup.log 2>&1
  cat .cuda_setup.log | tee -a "$LOG"
else
  echo "  setup_cuda_libpath.sh tidak ditemukan, lanjut dengan environment saat ini." | tee -a "$LOG"
fi

echo -e "\n[COMPILE]" | tee -a "$LOG"
gcc -O2 -o jacobi_sequential jacobi_sequential.c -lm && echo "  ✓ jacobi_sequential" | tee -a "$LOG"
mpicc -O2 -o jacobi_mpi jacobi_mpi.c -lm && echo "  ✓ jacobi_mpi" | tee -a "$LOG"
nvcc -O2 -allow-unsupported-compiler -o jacobi_cuda jacobi_cuda.cu && echo "  ✓ jacobi_cuda" | tee -a "$LOG"

echo -e "\n[A] SEQUENTIAL CPU" | tee -a "$LOG"
for N in $N_LIST; do
  echo -e "\n  --- N=$N ---" | tee -a "$LOG"
  ./jacobi_sequential "$N" | tee -a "$LOG"
done

echo -e "\n[B] MPI MULTICORE" | tee -a "$LOG"
for N in $N_LIST; do
  for NP in $MPI_NP_LIST; do
    echo -e "\n  --- N=$N  NP=$NP ---" | tee -a "$LOG"
    OMPI_MCA_coll_hcoll_enable=0 mpirun --allow-run-as-root -np "$NP" ./jacobi_mpi "$N" | tee -a "$LOG"
  done
done

echo -e "\n[C] CUDA" | tee -a "$LOG"
for N in $N_LIST; do
  for BS in $CUDA_BLOCK_LIST; do
    echo -e "\n  --- N=$N  BlockSize=$BS ---" | tee -a "$LOG"
    ./jacobi_cuda "$N" "$BS" | tee -a "$LOG"
  done
done

echo -e "\n==================================================" | tee -a "$LOG"
echo "  SELESAI. Hasil disimpan di: $LOG" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
