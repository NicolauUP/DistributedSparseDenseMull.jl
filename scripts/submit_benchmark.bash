#!/bin/bash


# Teste of N = 50k matrix
export BENCH_N=50000

for WORKERS in 8 16 32 64 127
do
    TASKS=$((WORKERS + 1))


    echo "Submitting benchmark with $WORKERS workers and matrix size $BENCH_N x $BENCH_N"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=slab_bench_${WORKERS}w_${BENCH_N}
#SBATCH --output=../logs/slab_bench_${WORKERS}w_${BENCH_N}.out
#SBATCH --ntasks=${TASKS}
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=normal-x86
#SBATCH -A f202409396cpcaa2x

julia --project=../. ../src/distributed_run.jl
EOT

done 