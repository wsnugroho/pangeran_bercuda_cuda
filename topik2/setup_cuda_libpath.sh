#!/bin/bash
# setup_cuda_libpath.sh
# Memastikan runtime cuBLAS/cuBLASLt CUDA 10.2 masuk ke LD_LIBRARY_PATH.
# Pakai: source ./setup_cuda_libpath.sh

add_libdir() {
    local d="$1"
    if [ -d "$d" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$d:"*) ;;
            *) export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}" ;;
        esac
    fi
}

has_libcublaslt10() {
    ldconfig -p 2>/dev/null | grep -q 'libcublasLt.so.10' && return 0
    local old_ifs="$IFS"
    IFS=':'
    for d in ${LD_LIBRARY_PATH:-}; do
        if ls "$d"/libcublasLt.so.10* >/dev/null 2>&1; then
            IFS="$old_ifs"
            return 0
        fi
    done
    IFS="$old_ifs"
    return 1
}

# Lokasi umum CUDA 10.2 dan NVIDIA HPC SDK.
for d in \
    /usr/local/cuda/lib64 \
    /usr/local/cuda-10.2/lib64 \
    /opt/cuda/lib64 \
    /opt/cuda-10.2/lib64 \
    /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/10.2/lib64 \
    /opt/nvidia/hpc_sdk/Linux_x86_64/*/math_libs/10.2/lib64
    do
        [ -e "$d" ] && add_libdir "$d"
    done

if ! has_libcublaslt10; then
    echo "[WARN] libcublasLt.so.10 belum ditemukan."
    echo "       Cari manual: find /usr/local /opt/nvidia -name 'libcublasLt.so.10*' 2>/dev/null"
    echo "       Lalu export LD_LIBRARY_PATH=/path/yang/berisi/library:\$LD_LIBRARY_PATH"
fi
