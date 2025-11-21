#!/bin/bash

export BENCH_N=50000
export ARCH_NAME="AMD_EPYC_7742"

# Create CsV Header if it does not exist
if [ ! -f "../results/architecture_benchmark_results.csv" ]; then
    echo "Architecture, N, Workers, Threads, Time(s), GFlops, Time Setup H (s), Time Setup X (s), Time Gather (s), Min Compute Time (s), Max Compute Time(s), Pct Imbalance (%)" > ../results/architecture_benchmark_results.csv
fi

# --- First Teste: Pure MPI --- #
for WORKERS in 4 8 16 32 64; do 
    TASKS=$((WORKERS + 1)) # +1 for master
    echo "Submitting architecture benchmark (Pure MPI) with $WORKERS workers and matrix size $BENCH_N x $BENCH_N"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=arch_bench_mpi_${WORKERS}w_${BENCH_N}
#SBATCH --output=../logs/arch_bench_mpi_${WORKERS}w_${BENCH_N}.out
#SBATCH --nodes=1
#SBATCH --ntasks=${TASKS}
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=normal-x86
#SBATCH -A f202409396cpcaa2x   
#SBATCH --mem=128G

julia --project=../. ../src/bench_distributed_run.jl 
EOT

done

# --- Second Teste: Hybrid MPI + Threads --- #
# Sweet Spot for EPYC 
echo "Starting Hybrid MPI + Threads Benchmarks"

submit_hybrid_benchmark() {
    local WORKERS=$1
    local THREADS=$2
    local TASKS=$((WORKERS + 1)) # +1 for master

    echo "Submitting architecture benchmark (Hybrid MPI + Threads) with $WORKERS workers, $THREADS threads and matrix size $BENCH_N x $BENCH_N"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=arch_bench_hybrid_${WORKERS}w_${THREADS}t_${BENCH_N}
#SBATCH --output=../logs/arch_bench_hybrid_${WORKERS}w_${THREADS}t_${BENCH_N}.out
#SBATCH --nodes=1
#SBATCH --ntasks=${TASKS}
#SBATCH --cpus-per-task=${THREADS}
#SBATCH --time=02:00:00
#SBATCH --partition=normal-x86
#SBATCH -A f202409396cpcaa2x   
#SBATCH --mem=128G

export SLURM_CPUS_PER_TASK=${THREADS}
julia --project=../. ../src/bench_distributed_run.jl 
EOT
}

# --- Battery of tests

# 1. Bus Saturation
submit_hybrid_benchmark 32 2 

# 2. Theoretical maximum 
submit_hybrid_benchmark 8 8 

# 3. Middle Ground
submit_hybrid_benchmark 16 4

# 4. Fat 
submit_hybrid_benchmark 4 16

# 5. Limit (pure threading)
submit_hybrid_benchmark 1 64