#!/bin/bash

#SBATCH --job-name=benchmark_thread_fil
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --mem=128G
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=../logs/benchmark_thread_fil_%j.out
#SBATCH --partition=normal-x86
#SBATCH -A f202409396cpcaa2x   

export JULIA_NUM_THREADS=128

echo "--- NUMA Topology ---"
numactl --show
echo "---------------------"


julia --project=../. ../src/mpi_transposed/setup_benchmark.jl -O3 