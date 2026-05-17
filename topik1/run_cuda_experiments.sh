#!/usr/bin/env bash
set -euo pipefail

nvcc -allow-unsupported-compiler experiment_a.cu -o experiment_a
nvcc -allow-unsupported-compiler experiment_increment.cu -o experiment_increment

echo ""
echo "=== Jalankan di GPU ==="
./experiment_a > output_experiment_a.txt
./experiment_increment > output_increment.txt

echo "Selesai. File output:"
echo "  output_experiment_a.txt"
echo "  output_increment.txt"
